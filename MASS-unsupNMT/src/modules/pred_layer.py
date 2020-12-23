import torch
import torch.nn.functional as F

class FixedPredLayer(torch.nn.Module):
    """
    Prediction layer, fix weight, not bias
    """
    def __init__(self, embeddings):
        super().__init__()
        self.n_words = embeddings.weight.size(0)
        dim = embeddings.weight.size(1)
        self.proj = torch.nn.Linear(dim, self.n_words, bias=False)
        self.proj.weight.data = embeddings.weight.data
        self.proj.requires_grad_(False)

    def forward(self, x, y):
        """

        Args:
            x: [n, dim]
            y: [n]

        Returns:

        """
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores, y, reduction='none')

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert x.dim() == 2
        return self.proj(x)
