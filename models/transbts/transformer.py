import torch.nn as nn
from models.transbts.intmd_sequential import IntmdSequential
from models.modules import ResBlock, PreNorm, PreNormDrop, FeedForward
from models.vit.blocks import SelfAttention


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = nn.ModuleList()
        for _ in range(depth):
            layers.append(
                ResBlock(
                    PreNormDrop(
                        SelfAttention(dim, num_heads=num_heads, dropout_rate=attn_dropout_rate),
                        dim,
                        dropout_rate
                    )
                )
            )
            layers.append(
                ResBlock(
                    PreNorm(FeedForward(dim, dropout_rate, hidden_dim=mlp_dim), dim)
                )
            )
        self.net = IntmdSequential(*layers)


    def forward(self, x):
        return self.net(x)
