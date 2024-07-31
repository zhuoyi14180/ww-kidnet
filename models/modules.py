import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x
    

class PreNorm(nn.Module):
    def __init__(self, module, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x):
        return self.module(self.norm(x))
    

class PreNormDrop(nn.Module):
    def __init__(self, module, dim, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.module = module

    def forward(self, x):
        return self.dropout(self.module(self.norm(x)))
    

class FeedForward(nn.Module):
    def __init__(self, in_dim, dropout_rate, hidden_dim=None, out_dim=None, act=nn.GELU):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
    