import torch
import torch.nn as nn
from models.utils.consts import WeightInit


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length=512):
        super().__init__()

        pe = torch.zeros(seq_length, embedding_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, seq_length, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x): # (1, 4096, 512)
        x = x + self.pe[:, :x.size(1), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) # 8x -> (1, 4096, 512)
        self.init_weights()

    
    def init_weights(self):
        nn.init.xavier_uniform_(self.pe)

    
    def forward(self, x):
        return x + self.pe
