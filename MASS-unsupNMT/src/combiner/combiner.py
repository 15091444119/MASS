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

    def forward(self, embeddings, lengths):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (max_length, batch_size)
            lengths: torch.LongTensor (batch_size)
        returns:
            outputs: torch.FloatTensor, (max_length, batch_size)
        """
        assert embeddings.size(1) == lengths.size(0)
        max_length = embeddings.size(0)
        # src_key_padding_mask set padding with false
        padding_mask = (~(get_masks(slen=max_length, lengths=lengths, causal=False)[0])).to(embeddings.device)  # (batch_size, max_length)
        outputs = self._transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)

        rep = self._context2rep(outputs.transpose(0, 1), lengths)
        rep = self._linear2(self._act(self._linear1(rep)))

        return rep


class GRUCombiner(nn.Module):
    def __init__(self, params):
        super().__init__()
        self._gru = nn.GRU(input_size=params.emb_dim, hidden_size=params.emb_dim, num_layers=params.n_combiner_layers, bidirectional=False, batch_first=False)
        self._context2rep = Context2Sentence(params.combiner_context_extractor)
        self._linear1 = nn.Linear(params.emb_dim, params.emb_dim)
        self._linear2 = nn.Linear(params.emb_dim, params.emb_dim)
        self._act = torch.nn.ReLU()

    def forward(self, embeddings, lengths):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (max_length, batch_size)
            lengths: torch.LongTensor (batch_size)
        returns:
            outputs: torch.FloatTensor, (batch_size, emb_dim)
        """
        output, _ = self._gru(embeddings)
        rep = self._context2rep(output.transpose(0, 1), lengths)
        rep = self._linear2(self._act(self._linear1(rep)))

        return rep


def build_combiner(params):
    if params.combiner == "transformer":
        return TransformerCombiner(params)
    elif params.combiner == "gru":
        return GRUCombiner(params)
    else:
        raise NotImplementedError


class MultiLingualCombiner(nn.Module):

    def __init__(self, params):
        super().__init__()
        if params.share_combiner:
            combiner = build_combiner(params)
            self._lang2combiner = nn.ModuleDict({
                lang: combiner for lang in params.combiner_steps
            })
        else:
            self._lang2combiner = nn.ModuleDict({
                lang: build_combiner(params) for lang in params.combiner_steps
            })

    def forward(self, embeddings, lengths, lang):
        combiner = self._lang2combiner[lang]
        return combiner(embeddings, lengths)

