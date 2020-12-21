import torch.nn as nn
import torch
from src.model.transformer import get_masks

from src.model.combiner.context_combiner.constant import COMBINE_END, COMBINE_FRONT
from src.model.transformer import create_sinusoidal_embeddings, N_MAX_POSITIONS, Embedding


def check_combiner_inputs(encoded, lengths, combiner_labels):
    bs, slen, dim = encoded.size()
    assert bs == lengths.size(0)
    assert slen == lengths.max()
    assert (bs, slen) == combiner_labels.size()


class Combiner(nn.Module):
    """
    Combine representations
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        """
        Combine subword representations to whole word representation
        combine subword front to subword end into whole word representation
        """
        return self.combine(*args, **kwargs)

    def combine(self, *args, **kwargs):
        raise NotImplementedError


class AverageCombiner(Combiner):
    """
    Use the average of tokens of the mass encoded representation as the word representation
    """
    def __init__(self):
        super().__init__()

    def combine(self, encoded, lengths, combine_labels):
        """
        Args:
            encoded: [bs, len, dim]
            lengths: [bs]
                lengths of encoded tokens(not final length)
            combine_labels: [bs, len]

        Returns:
            representation: [splitted word number, dim]
            trained_representation
        """
        check_combiner_inputs(encoded, lengths, combine_labels)
        bs, max_len, dim = encoded.size()
        representations = []
        for i in range(bs):
            token_id = 0
            while(token_id < max_len): #
                if combine_labels[i][token_id] == COMBINE_FRONT:
                    front_id = token_id
                    while(combine_labels[i][token_id] != COMBINE_END):
                        token_id += 1
                        assert token_id < max_len
                    representations.append(encoded[i][front_id:token_id + 1].mean(dim=0))
                token_id += 1
        return torch.stack(representations, dim=0)


class LastTokenCombiner(Combiner):
    """
    Use a transformer layer above mass encoder, and use the last token of each word for representation
    """

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.n_heads,
                                                       dim_feedforward=params.emb_dim * 4)

        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers)
        self.output_dim = params.emb_dim

    def combine(self, encoded, lengths, combine_labels, lang_id):
        """
        Args:
            encoded: [bs, len, dim]
            lengths: [bs]
            combine_labels: [bs, len]
            lang_id: int
                language index

        Returns:
            representation: [splitted word number, dim]
            trained_representation

        """
        check_combiner_inputs(encoded, lengths, combine_labels)

        transformer_encoder = self.encoder
        encoded = encoded.transpose(0, 1)  #[len, bs, dim]
        max_length = encoded.size(0)

        subword_last_token_mask = self.word_end_mask(combine_labels).unsqueeze(-1) # [bs, len, 1]

        # nothing to combine
        if subword_last_token_mask.to(torch.long).sum() == 0:
            return None

        # src_key_padding_mask set padding with true
        padding_mask = get_torch_transformer_encoder_mask(lengths=lengths).to(encoded.device)
        outputs = transformer_encoder(src=encoded, src_key_padding_mask=padding_mask) # [len, bs, dim]
        outputs = outputs.transpose(0, 1) # [bs, len, dim]

        representation = outputs.masked_select(subword_last_token_mask).view(-1, self.output_dim)

        return representation

    @classmethod
    def word_end_mask(cls, combine_labels):
        mask = combine_labels.eq(COMBINE_END)
        return mask


def get_torch_transformer_encoder_mask(lengths):
    """
    Args:
        lengths: [bs]

    Returns:

    """
    slen = lengths.max().item()
    return (~(get_masks(slen=slen, lengths=lengths, causal=False)[0]))  # (batch_size, max_length)


class WordInputCombiner(Combiner):

    def __init__(self, params):
        super().__init__()
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2

        self.position_embeddings = Embedding(N_MAX_POSITIONS, params.emb_dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, params.emb_dim, out=self.position_embeddings.weight)

        self.special_embeddings = Embedding(3, params.emb_dim, padding_idx=self.pad_id)  # bos and eos embedding for word combiner

        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.n_heads,
                                                       dim_feedforward=params.emb_dim * 4)

        self.another_context_encoder = \
            nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers)

        self.word_combiner =  \
            nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers)

    def another_encode(self, encoded, lengths, lang_id):
        """

        Args:
            encoded:  [bs, len, dim]
            lengths: [dim]
            lang_id:  int

        Returns:

        """
        another_context_encoder = self.another_context_encoder
        bs, slen, _ = encoded.size()
        positions = torch.arange(slen).to(encoded.device).long().unsqueeze(-1)  # [len, 1]
        encoded = encoded.transpose(0, 1)

        positional_embeddings = self.position_embeddings(positions).expand_as(encoded)  # [len, bs, dim]

        mask = get_torch_transformer_encoder_mask(lengths)  # [bs, len]

        encoded = another_context_encoder(src=encoded + positional_embeddings, src_key_padding_mask=mask)  # [len, bs, dim]

        return encoded, lengths

    def word_encode(self, word_combiner_inputs, word_combiner_lengths, lang_id):
        """

        Args:
            word_combiner_inputs: [len, bs, dim]
            word_combiner_lengths: [len]
            lang_id: int

        Returns:

        """
        word_combiner = self.word_combiner
        slen = word_combiner_lengths.max()
        dim = word_combiner_inputs.size(-1)

        positions = torch.arange(slen).to(word_combiner_inputs.device).long().unsqueeze(-1)  # [len, 1]
        positional_embeddings = self.position_embeddings(positions).expand_as(word_combiner_inputs)  # [len, bs, dim]

        word_combiner_mask = get_torch_transformer_encoder_mask(word_combiner_lengths)

        representation = word_combiner(src=word_combiner_inputs + positional_embeddings, src_key_padding_mask=word_combiner_mask)[0].view(-1, dim)  # use bos index as word representation

        return representation

    def combine(self, encoded, lengths, combine_labels, lang_id):
        if (combine_labels == COMBINE_END).long().sum() == 0:
            return None

        check_combiner_inputs(encoded, lengths, combine_labels)

        encoded, lengths = self.another_encode(encoded, lengths, lang_id)
        word_combiner_inputs, word_combiner_lengths = self.get_word_combiner_inputs(
            encoded=encoded,
            lengths=lengths,
            combine_labels=combine_labels
        )  # [len, bs, dim]

        representation = self.word_encode(word_combiner_inputs, word_combiner_lengths, lang_id)

        return representation

    def get_word_combiner_inputs(self, encoded, lengths, combine_labels):
        """

        Args:
            encoded: [len, bs, dim]
            lengths:  [bs]
            combine_labels: [bs, len]
        Returns:
        """
        slen, bs, dim = encoded.size()

        # calculate max word length
        word_combiner_lengths = []
        for batch_id in range(bs):
            cur_word_length = 0
            for token_id in range(slen):
                if combine_labels[batch_id][token_id] == COMBINE_FRONT:
                    cur_word_length += 1
                elif combine_labels[batch_id][token_id] == COMBINE_END:
                    cur_word_length += 1
                    word_combiner_lengths.append(cur_word_length)
                    cur_word_length = 0
                else:
                    cur_word_length = 0

        word_combiner_lengths = torch.tensor(word_combiner_lengths).long().to(lengths.device) + 2  # 2 means bos and eos
        max_word_length = word_combiner_lengths.max().item()

        # word number
        n_word = word_combiner_lengths.size(0)

        # selecting_mask
        to_token_mask = torch.BoolTensor(n_word, max_word_length).fill_(False).to(encoded.device)
        from_token_mask = torch.BoolTensor(bs, slen).fill_(False).to(encoded.device)

        eos_mask = torch.BoolTensor(n_word, max_word_length).fill_(False).to(encoded.device)

        word_id = 0
        for batch_id in range(bs):
            cur_word_length = 0
            for token_id in range(slen):
                if combine_labels[batch_id][token_id] == COMBINE_FRONT:
                    to_token_mask[word_id][1 + cur_word_length] = True
                    from_token_mask[batch_id][token_id] = True

                    cur_word_length += 1
                elif combine_labels[batch_id][token_id] == COMBINE_END:
                    to_token_mask[word_id][1 + cur_word_length] = True
                    eos_mask[word_id][2 + cur_word_length] = True
                    from_token_mask[batch_id][token_id] = True

                    cur_word_length = 0

                    word_id += 1
                else:
                    cur_word_length = 0

        # create input embeddings
        word_combiner_inputs = torch.FloatTensor(n_word, max_word_length, dim).to(encoded.device).fill_(0.0)

        # bos
        word_combiner_inputs[:, 0] = self.special_embeddings(torch.tensor(self.bos_id).long().to(encoded.device))

        # eos
        word_combiner_inputs[eos_mask.unsqueeze(-1).expand_as(word_combiner_inputs)] = self.special_embeddings(torch.tensor([self.eos_id] * n_word).long().to(encoded.device)).view(-1)

        encoded = encoded.transpose(0, 1) #[bs, len, dim]
        # words
        word_combiner_inputs[to_token_mask.unsqueeze(-1).expand_as(word_combiner_inputs)] = encoded[from_token_mask.unsqueeze(-1).expand_as(encoded)]

        return word_combiner_inputs.transpose(0, 1), word_combiner_lengths


def build_combiner(params):
    if params.combiner == "last_token":
        return LastTokenCombiner(params)
    elif params.combiner == "average":
        return AverageCombiner()
    elif params.combiner == "word_input":
        return WordInputCombiner(params)
    else:
        raise Exception("No combiner named: {}".format(params.combiner))

