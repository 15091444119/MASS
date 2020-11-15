import torch.nn as nn
import torch
import pdb
from src.model.transformer import get_masks
from src.evaluation.utils import Context2Sentence

from src.utils import AttrDict
from src.evaluation.utils import package_module
from .constant import SKIPPED_TOKEN, SUBWORD_END, SUBWORD_FRONT, NOT_USED_TOKEN, MASKED_TOKEN


class Combiner(nn.Module):
    """
    Combine representations
    """

    def __init__(self):
        super().__init__()

    def forward(self, method, *args, **kwargs):
        """
        Combine subword representations to whole word representation
        combine subword front to subword end into whole word representation
        """
        if method == "combine":
            return self.combine(*args, **kwargs)
        elif method == "encode":
            return self.encode(*args, **kwargs)
        else:
            raise NotImplementedError

    def combine(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, encoded, lengths, combine_labels, lang, trained_combine_labels=None):
        """
        combine subwords
        Params:
            encoded: [bs, len, dim]
            lengths: [bs]
            combine_labels: [bs, len]
            lang: language
            trained_combine_labels: combine labels for self-training
        combine, and get the new representation
        if combine label is skipped, use the original representation
        else we use the combine function to generate a whole word representation

        masked tokens are also counted as tokens, but a mask_mask will be returned

        Returns:
            new_encoded [bs, new_len, dim], new_lens [bs, new_len, dim]
        """
        representation, trained_representation = self.combine(encoded, lengths, combine_labels, lang, trained_combine_labels) # [split word number, dim]

        # get length
        new_lens = []
        for sentence_labels in combine_labels:
            cur_len = 0
            for label in sentence_labels:
                label = label.item()
                if label == SKIPPED_TOKEN or label == SUBWORD_END or label == MASKED_TOKEN:
                    cur_len += 1
            assert cur_len == (sentence_labels.eq(SKIPPED_TOKEN).sum() + sentence_labels.eq(SUBWORD_END).sum() + sentence_labels.eq(MASKED_TOKEN).sum())
            new_lens.append(cur_len)

        cur_rep = 0
        bs, _, dim = encoded.size()
        new_encoded = torch.FloatTensor(bs, max(new_lens), dim).fill_(0.0).cuda()
        mask_mask = torch.BoolTensor(bs, max(new_lens)).fill_(False).cuda()  # masked tokens will be set to True
        for sentence_idx, sentence_labels in enumerate(combine_labels):
            cur_pos = 0
            for token_idx, label in enumerate(sentence_labels):
                label = label.item()
                if label == SKIPPED_TOKEN or label == MASKED_TOKEN:
                    new_encoded[sentence_idx][cur_pos] = encoded[sentence_idx][token_idx]
                    if label == MASKED_TOKEN:
                        mask_mask[sentence_idx][cur_pos] = True
                    cur_pos += 1
                elif label == SUBWORD_END:
                    new_encoded[sentence_idx][cur_pos] = representation[cur_rep]
                    cur_pos += 1
                    cur_rep += 1

        new_lens = torch.LongTensor(new_lens).cuda()
        assert cur_rep == len(representation)  # all representations are used

        return new_encoded, new_lens, mask_mask, trained_representation


class LastTokenCombiner(Combiner):

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.n_heads,
                                                       dim_feedforward=params.emb_dim * 4)

        self.encoder = torch.nn.ModuleDict(
            {lang: nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers) for lang in params.lang2id}
        )
        self.output_dim = params.emb_dim

    def combine(self, encoded, lengths, combine_labels, lang, trained_combine_labels=None):
        """
        Args:
            encoded: [bs, len, dim]
            lengths: [bs]
            combine_labels: [bs, len]
            lang_id: int
                language index
            trained_combine_labels: another label

        Returns:
            representation: [splitted word number, dim]
            trained_representation

        """
        transformer_encoder = self.encoder[lang]
        encoded = encoded.transpose(0, 1)
        assert encoded.size(1) == lengths.size(0)
        max_length = encoded.size(0)

        subword_last_token_mask = self.word_end_mask(combine_labels).unsqueeze(-1) # [bs, len, 1]

        # nothing to combine
        if subword_last_token_mask.to(torch.long).sum() == 0:
            return [], None

        # src_key_padding_mask set padding with false
        padding_mask = (~(get_masks(slen=max_length, lengths=lengths, causal=False)[0])).to(
            encoded.device)  # (batch_size, max_length)
        outputs = transformer_encoder(src=encoded, src_key_padding_mask=padding_mask) #[len, bs, dim]
        outputs = outputs.transpose(0, 1) #[bs, len, dim]

        representation = outputs.masked_select(subword_last_token_mask).view(-1, self.output_dim)

        if trained_combine_labels is not None:
            trained_subword_last_token_mask = self.word_end_mask(trained_combine_labels).unsqueeze(-1) # [bs, len, 1]
            trained_representation = outputs.masked_select(trained_subword_last_token_mask).view(-1, self.output_dim)
            return representation, trained_representation
        else:
            return representation, None

    @classmethod
    def word_end_mask(cls, combine_labels):
        mask = combine_labels.eq(SUBWORD_END)
        return mask


class TransformerCombiner(nn.Module):

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.n_heads,
                                                       dim_feedforward=params.emb_dim * 4)
        self._transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers)
        self._context2rep = Context2Sentence(params.combiner_context_extractor)
        self._linear1 = nn.Linear(params.emb_dim, params.emb_dim)
        self._linear2 = nn.Linear(params.emb_dim, params.emb_dim)
        self._act = torch.nn.ReLU()

    def forward(self, embeddings, lengths, lang):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (max_length, batch_size)
            lengths: torch.LongTensor (batch_size)
            lang: not used
        returns:
            outputs: torch.FloatTensor, (batch_size, emb_dim)
        """
        assert embeddings.size(1) == lengths.size(0)
        max_length = embeddings.size(0)
        # src_key_padding_mask set padding with false
        padding_mask = (~(get_masks(slen=max_length, lengths=lengths, causal=False)[0])).to(embeddings.device)  # (batch_size, max_length)
        outputs = self._transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)

        rep = self._context2rep(outputs.transpose(0, 1), lengths)
        rep = self._linear2(self._act(self._linear1(rep)))

        return rep


class MultiLingualNoneParaCombiner(nn.Module):
    def __init__(self, method):
        super().__init__()
        self._context2sentence = Context2Sentence(method)

    def forward(self, embeddings, lengths, lang):
        """
        params
            embeddings: torch.FloatTensor (max_length, batch_size)

            lengths: torch.FloatTensor (batch_size)

            lang: list of strings

        Returns:
            rep
        """
        return self._context2sentence(embeddings.transpose(0, 1), lengths)


class GRUCombiner(nn.Module):
    def __init__(self, params):
        super().__init__()
        self._gru = nn.GRU(input_size=params.emb_dim, hidden_size=params.emb_dim, num_layers=params.n_combiner_layers, bidirectional=False, batch_first=False)
        self._context2rep = Context2Sentence(params.combiner_context_extractor)
        self._linear1 = nn.Linear(params.emb_dim, params.emb_dim)
        self._linear2 = nn.Linear(params.emb_dim, params.emb_dim)
        self._act = torch.nn.ReLU()

    def forward(self, embeddings, lengths, lang=None):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (max_length, batch_size)
            lengths: torch.LongTensor (batch_size)
            lang: interface para, not used
        returns:
            outputs: torch.FloatTensor, (batch_size, emb_dim)
        """
        assert embeddings.size(1) == lengths.size(0)
        output, _ = self._gru(embeddings)
        rep = self._context2rep(output.transpose(0, 1), lengths)
        rep = self._linear2(self._act(self._linear1(rep)))

        return rep


class LinearCombiner(nn.Module):
    """
    context2rep, then linear
    """

    def __init__(self, params):
        super().__init__()
        self._context2rep = Context2Sentence(params.combiner_context_extractor)
        self._linear = nn.Linear(params.emb_dim, params.emb_dim)

    def forward(self, embeddings, lengths, lang=None):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (max_length, batch_size)
            lengths: torch.LongTensor (batch_size)
            lang: not used
        returns:
            outputs: torch.FloatTensor, (batch_size, emb_dim)
        """
        assert embeddings.size(1) == lengths.size(0)
        rep = self._context2rep(embeddings.transpose(0, 1), lengths)
        rep = self._linear(rep)

        return rep


def build_combiner(params):
    return LastTokenCombiner(params)


class BiLingualCombiner(nn.Module):

    def __init__(self, src_lang, src_combiner, tgt_lang, tgt_combiner):
        super().__init__()
        self._lang2combiner = nn.ModuleDict({
            src_lang: src_combiner, tgt_lang: tgt_combiner
        })

    def forward(self, embeddings, lengths, lang):
        combiner = self._lang2combiner[lang]
        return combiner(embeddings, lengths)


def load_combiner_model(model_path):
    reloaded = torch.load(model_path)
    model_params = AttrDict(reloaded['params'])
    combiner = build_combiner(model_params).cuda()

    combiner.load_state_dict(package_module(reloaded["combiner"]))

    return model_params, combiner
