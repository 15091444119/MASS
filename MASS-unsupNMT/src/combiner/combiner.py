import torch.nn as nn
import torch
from src.model.transformer import get_masks

class Transformer(nn.Module):

    def __init__(self, params):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=params.emb_dim, nhead=params.num_head,
                                                       dim_feedforward=params.hidden_dim)
        self._transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=params.num_layers)

    def forward(self, embeddings, lengths):
        padding_mask = ~get_mask(lengths, all_words=True, batch_first=True, cuda=embeddings.is_cuda)
        embeddings = embeddings.transpose(0, 1)  # not batch first
        outputs = self._transformer_encoder(src=embeddings, src_key_padding_mask=padding_mask)
        outputs = outputs.transpose(0, 1)
        return outputs
    def build_combiner(params):
