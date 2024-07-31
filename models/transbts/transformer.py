import torch.nn as nn
from models.transbts.intmd_sequential import IntmdSequential
from models.modules import ResBlock, PreNorm, PreNormDrop, FeedForward


class SelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D) -> (B, N, H, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
