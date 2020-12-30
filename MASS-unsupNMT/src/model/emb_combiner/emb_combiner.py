import torch
import torch.nn as nn
from src.modules.context_extractor import Context2Sentence
from src.modules.encoders import TransformerEncoder
from src.modules.mlp import MLP


class BaseEmbCombiner(torch.nn.Module):
    """
    abstract class for embedding combiner
    """

    def __init__(self):
        super().__init__()

    def forward(self, embeddings, lengths):
        """

        Args:
            embeddings: [bs, slen, dim]
            lengths: [bs]
        Returns:
            torch.LongTensor [bs, dim]
            combined representation of each word
        """
        raise NotImplementedError


class WordTokenAverage(BaseEmbCombiner):

    def __init__(self):
        super().__init__()
        self.context2rep = Context2Sentence("word_token_average")
        self.linear = nn.Linear(2, 2)  # thie parametar is not used

    def forward(self, embeddings, lengths):
        rep = self.context2rep(embeddings, lengths)
        return rep


class LinearEmbCombiner(BaseEmbCombiner):
    """
    context2rep, then linear
    """

    def __init__(self, emb_dim, context_extractor):
        super().__init__()
        self.context2rep = Context2Sentence(context_extractor)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, embeddings, lengths):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (bs, slen, dim)
            lengths: torch.LongTensor (bs)
        returns:
            outputs: torch.FloatTensor, (bs, dim)
        """
        rep = self.context2rep(embeddings, lengths)
        rep = self.linear(rep)

        return rep


class GRUEmbCombiner(nn.Module):
    def __init__(self, emb_dim, n_layer, context_extractor):
        super().__init__()
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=emb_dim, num_layers=n_layer, bidirectional=False, batch_first=True)
        self.context2rep = Context2Sentence(context_extractor)
        self.mlp = MLP(emb_dim)

    def forward(self, embeddings, lengths):
        """
        forward function of the combiner
        params:
            embeddings: torch.FloatTensor, (bs, slen, dim)
            lengths: torch.LongTensor (bs, len)
        returns:
            outputs: torch.FloatTensor, (bs, dim)
        """
        output, _ = self.gru(embeddings)

        rep = self.context2rep(output, lengths)

        rep = self.mlp(rep)

        return rep


class TransformerCombiner(BaseEmbCombiner):
    """
    Use a transformer layer above mass encoder, and use the last token of each word for representation
    """

    def __init__(self, emb_dim, n_layer, n_head, context_extractor):
        super().__init__()

        self.encoder = TransformerEncoder(emb_dim, n_layer, n_head)
        self.context2rep = Context2Sentence(context_extractor)
        self.mlp = MLP(emb_dim)

    def forward(self, embeddings, lengths):
        """

        Args:
            embeddings: [bs, len, dim]
            lengths: [bs]

        Returns:
            representation: [splitted word number, dim]
            trained_representation
        """

        encoded = self.encoder(embeddings, lengths)

        sentence_rep = self.context2rep(encoded, lengths)

        rep = self.mlp(sentence_rep)

        return rep
