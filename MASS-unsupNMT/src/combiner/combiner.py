import torch.nn as nn
import torch
import pdb
from src.model.transformer import get_masks
from src.evaluation.utils import Context2Sentence


class TransformerCombiner(nn.Module):

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.n_heads,
                                                       dim_feedforward=params.emb_dim * 4)
        self._transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers)
        self._context2rep = Context2Sentence(params.combiner_context2sentence)
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
    if params.combiner == "transformer":
        return TransformerCombiner(params)
    elif params.combiner == "gru":
        return GRUCombiner(params)
    elif params.combiner == "linear":
        return LinearCombiner(params)
    else:
        raise NotImplementedError


class BiLingualCombiner(nn.Module):

    def __init__(self, src_lang, src_combiner, tgt_lang, tgt_combiner):
        super().__init__()
        self._lang2combiner = nn.ModuleDict({
            src_lang: src_combiner, tgt_lang: tgt_combiner
        })

    def forward(self, embeddings, lengths, lang):
        combiner = self._lang2combiner[lang]
        return combiner(embeddings, lengths)
