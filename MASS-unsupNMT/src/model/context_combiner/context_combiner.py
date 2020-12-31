import torch.nn as nn
import torch
from src.model.transformer import get_masks

from src.model.context_combiner.constant import COMBINE_END, COMBINE_FRONT
from src.modules.encoders import TransformerEncoder
from src.model.transformer import Embedding
from src.utils import AttrDict


def check_combiner_inputs(encoded, lengths, combiner_labels):
    bs, slen, dim = encoded.size()
    assert bs == lengths.size(0)
    #assert slen == lengths.max()
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

    def __init__(self, emb_dim, sinusoidal_embeddings, n_head, n_layer):
        super().__init__()
        self.encoder = TransformerEncoder(emb_dim=emb_dim, sinusoidal_embeddings=sinusoidal_embeddings, n_head=n_head, n_layer=n_layer)
        self.emb_dim = emb_dim
        self.combine_label_embeddings = Embedding(5, embedding_dim=emb_dim)

    def combine(self, encoded, lengths, combine_labels):
        """
        Args:
            encoded: [bs, len, dim]
            lengths: [bs]
            combine_labels: [bs, len]

        Returns:
            representation: [splitted word number, dim]
            trained_representation

        """
        check_combiner_inputs(encoded, lengths, combine_labels)

        encoded = encoded + self.combine_label_embeddings(combine_labels + 1)

        subword_last_token_mask = self.word_end_mask(combine_labels).unsqueeze(-1) # [bs, len, 1]

        # nothing to combine
        if subword_last_token_mask.to(torch.long).sum() == 0:
            return None

        outputs = self.encoder(encoded, lengths)

        representation = outputs.masked_select(subword_last_token_mask).view(-1, self.emb_dim)

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

    def __init__(self, emb_dim, sinusoidal_embeddings, n_head, n_another_context_encoder_layer, n_word_combiner_layer):
        super().__init__()
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2


        self.special_embeddings = Embedding(3, emb_dim, padding_idx=self.pad_id)  # bos and eos embedding for word combiner

        self.another_context_encoder = \
            TransformerEncoder(emb_dim=emb_dim, sinusoidal_embeddings=sinusoidal_embeddings, n_layer=n_another_context_encoder_layer, n_head=n_head)

        self.word_combiner =  \
            TransformerEncoder(emb_dim=emb_dim, sinusoidal_embeddings=sinusoidal_embeddings, n_layer=n_word_combiner_layer, n_head=n_head)

    def another_encode(self, encoded, lengths):
        """

        Args:
            encoded:  [bs, len, dim]
            lengths: bs]

        Returns:

        """
        encoded = self.another_context_encoder(encoded, lengths)

        return encoded, lengths

    def word_encode(self, word_combiner_inputs, word_combiner_lengths):
        """

        Args:
            word_combiner_inputs: [len, bs, dim]
            word_combiner_lengths: [len]

        Returns:

        """
        dim = word_combiner_inputs.size(-1)

        encoded = self.word_combiner(word_combiner_inputs, word_combiner_lengths)

        representation = encoded[:, 0, :].view(-1, dim)

        return representation

    def combine(self, encoded, lengths, combine_labels):
        if (combine_labels == COMBINE_END).long().sum() == 0:
            return None

        check_combiner_inputs(encoded, lengths, combine_labels)

        encoded, lengths = self.another_encode(encoded, lengths)

        word_combiner_inputs, word_combiner_lengths = self.get_word_combiner_inputs(
            encoded=encoded.transpose(0, 1),
            lengths=lengths,
            combine_labels=combine_labels
        )  # [bs, len, dim]

        representation = self.word_encode(word_combiner_inputs, word_combiner_lengths)

        return representation

    def get_word_combiner_inputs(self, encoded, lengths, combine_labels):
        """

        Args:
            encoded: [len, bs, dim]
            lengths:  [bs]
            combine_labels: [bs, len]
        Returns:
            tuple:
                word_combiner_inputs: [bs, len, dim]
                word_combiner_lengths: [bs]
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

        return word_combiner_inputs, word_combiner_lengths


def build_combiner(params):
    if params.combiner == "last_token":
        return LastTokenCombiner(emb_dim=params.emb_dim, n_head=params.n_head, n_layer=params.n_layer, sinusoidal_embeddings=params.sinusoidal_embeddings)
    elif params.combiner == "average":
        return AverageCombiner()
    elif params.combiner == "word_input":
        return WordInputCombiner(
            emb_dim=params.emb_dim,
            sinusoidal_embeddings=params.sinusoidal_embeddings,
            n_head=params.n_head,
            n_another_context_encoder_layer=params.n_another_context_encoder_layer,
            n_word_combiner_layer=params.n_word_combiner_layer
        )
    else:
        raise Exception("No combiner named: {}".format(params.combiner))


def load_combiner(path):
    reloaded = torch.load(path)

    combiner_train_params = AttrDict(reloaded["params"])
    combiner = build_combiner(combiner_train_params)

    combiner.load_state_dict(reloaded["combiner"])

    return combiner
