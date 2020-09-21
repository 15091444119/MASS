import torch.nn as nn
import torch
from src.model.transformer import get_masks


class Transformer(nn.Module):

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.num_head,
                                                       dim_feedforward=params.emb_dim * 4)
        self._transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=params.combiner_num_layers)

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
        padding_mask = ~get_masks(slen=max_length, lengths=lengths, causal=False)  # (batch_size, max_length)
        outputs = self._transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)
        return outputs
