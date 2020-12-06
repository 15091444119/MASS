import torch.nn as nn
import torch
import pdb
from src.model.transformer import get_masks
from src.evaluation.utils import Context2Sentence

from src.utils import AttrDict
from src.evaluation.utils import package_module
from src.combiner.constant import COMBINE_END, COMBINE_FRONT


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
        else:
            raise NotImplementedError

    def combine(self, *args, **kwargs):
        raise NotImplementedError


class AverageCombiner(Combiner):
    """
    Use the average of tokens of the mass encoded representation as the word representation
    """
    def __init__(self):
        super().__init__()

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

        self.encoder = torch.nn.ModuleList(
            [nn.TransformerEncoder(transformer_layer, num_layers=params.n_combiner_layers) for _ in params.lang2id.keys()]
        )
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
        transformer_encoder = self.encoder[lang_id]
        encoded = encoded.transpose(0, 1)  #[len, bs, dim]
        assert encoded.size(1) == lengths.size(0)
        max_length = encoded.size(0)

        subword_last_token_mask = self.word_end_mask(combine_labels).unsqueeze(-1) # [bs, len, 1]

        # nothing to combine
        if subword_last_token_mask.to(torch.long).sum() == 0:
            return None

        # src_key_padding_mask set padding with true
        padding_mask = (~(get_masks(slen=max_length, lengths=lengths, causal=False)[0])).to(
            encoded.device)  # (batch_size, max_length)
        outputs = transformer_encoder(src=encoded, src_key_padding_mask=padding_mask) # [len, bs, dim]
        outputs = outputs.transpose(0, 1) # [bs, len, dim]

        representation = outputs.masked_select(subword_last_token_mask).view(-1, self.output_dim)

        return representation

    @classmethod
    def word_end_mask(cls, combine_labels):
        mask = combine_labels.eq(COMBINE_END)
        return mask


def build_combiner(params):
    if params.combiner == "last_token":
        return LastTokenCombiner(params)
    elif params.combiner == "average":
        return AverageCombiner()
    else:
        raise Exception("No combiner named: {}".format(params.combiner))

