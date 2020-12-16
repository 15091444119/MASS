import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, emb_dim):

        super().__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.linear2 = nn.Linear(emb_dim, emb_dim)
        self.act = nn.ReLU()

    def forward(self, rep):

        return self.linear2(self.act(self.linear1(rep)))
