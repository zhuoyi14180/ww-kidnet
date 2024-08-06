import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, '/share/project/zhuoyi/ww-kidnet/models')

from vit.blocks import PatchEmbedding, DropPath, SelfAttention
from modules import PreNorm, FeedForward
from vit.blocks import SelfAttention


    

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio=4., dropout_rate=0.1, drop_path_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        attn = SelfAttention(embedding_dim, num_heads, dropout_rate=dropout_rate)
        mlp = FeedForward(embedding_dim, dropout_rate, hidden_dim=int(embedding_dim * mlp_ratio))
        self.norm1 = PreNorm(attn, embedding_dim)
        self.norm2 = PreNorm(mlp, embedding_dim)

    def forward(self, x):
        x = x + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(x))
        return x


class VisionTransformer2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=4, num_classes=4, embedding_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout_rate=0.1, cls=0, final_act=nn.Softmax):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pe = nn.Parameter(torch.zeros(1, num_patches + cls, embedding_dim))
        self.pe_drop = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, mlp_ratio, dropout_rate) for _ in range(depth)
            ]
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.deconv = nn.ConvTranspose2d(embedding_dim, num_classes, kernel_size=patch_size, stride=patch_size)
        self.cls = cls

        if cls:
            self.outc = nn.Linear(embedding_dim, num_classes)
        else:
            self.deconv = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=patch_size, stride=patch_size)
            self.outc = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        if final_act is not None:
            self.final_act = final_act(dim=1)
        else:
            self.final_act = None
        self._init_weights()


    def _init_weights(self):
        nn.init.trunc_normal_(self.pe, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)


    def _init_vit_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    
    def forward(self, x):
        if self.final_act == None:
            return self._cls_task(x) if self.cls else self._seg_task(x)
        return self.final_act(self._cls_task(x)) if self.cls else self.final_act(self._seg_task(x))
    

    def _seg_task(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)

        x = x + self.pe
        x = self.pe_drop(x)

        x = self.net(x)
        x = self.pre_head_ln(x)

        x = x.permute(0, 2, 1).contiguous()  # (B, embedding_dim, num_patches)
        x = x.view(B, self.embedding_dim, H // self.patch_size, W // self.patch_size)
        x = self.deconv(x)
        return self.outc(x)


    def _cls_task(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        x = x + self.pe
        x = self.pe_drop(x)

        x = self.net(x)
        x = self.pre_head_ln(x)

        x = x[:, 0]
        return self.outc(x)


def get_default():
    return VisionTransformer2D()


if __name__ == "__main__":

    batch_size = 8
    img_size = 224
    num_classes = 4
    in_channels = 4

    model = VisionTransformer2D(img_size=img_size, in_channels=in_channels, num_classes=num_classes, cls=0).cuda()
    x = torch.randn(batch_size, in_channels, img_size, img_size).cuda()
    output = model(x)
    print(output.shape)