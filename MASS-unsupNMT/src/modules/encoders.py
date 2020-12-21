import torch.nn as nn
import torch
from src.model.transformer import Embedding, N_MAX_POSITIONS, create_sinusoidal_embeddings, get_masks


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with positional embedding
    """

    def __init__(self, emb_dim, sinusoidal_embeddings, n_layer, n_head):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_head,
            dim_feedforward=emb_dim * 4
        )

        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=n_layer)
        self.position_embeddings = Embedding(N_MAX_POSITIONS, emb_dim)
        if sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, emb_dim, out=self.position_embeddings.weight)

    def forward(self, embeddings, lengths):
        """

        Args:
            embeddings: [bs, len, dim]
            lengths: [bs]

        Returns:
            [bs, len, dim]
        """
        bs, slen, _ = embeddings.size()

        positions = torch.arange(slen).to(embeddings.device).long().unsqueeze(-1)  # [len, 1]
        embeddings = embeddings.transpose(0, 1)

        positional_embeddings = self.position_embeddings(positions).expand_as(embeddings)  # [len, bs, dim]

        embeddings = embeddings + positional_embeddings

        mask = get_torch_transformer_encoder_mask(lengths)  # [bs, len]

        encoded = self.encoder(embeddings, src_key_padding_mask=mask).transpose(0, 1)

        return encoded


def get_torch_transformer_encoder_mask(lengths):
    """
    Args:
        lengths: [bs]

    Returns:

    """
    slen = lengths.max().item()
    return (~(get_masks(slen=slen, lengths=lengths, causal=False)[0]))  # (batch_size, max_length)


