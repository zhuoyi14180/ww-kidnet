import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super().__init__()

        pe = torch.zeros(max_length, embedding_dim) # (maxlength, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) # (max_length, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        ) # (embedding_dim / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # (max_length, embedding_dim / 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_length, 1, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, seq_length=4096, embedding_dim=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))

    def forward(self, x):
        return x + self.pe