import torch
import torch.nn as nn
from models.transbts.transformer import Transformer
from models.transbts.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
from models.transbts.unet import UNet


class TransBTS(UNet):
    def __init__(
            self, 
            img_dim, 
            patch_dim, 
            num_channels, 
            embedding_dim, 
            num_heads, 
            num_layers, 
            hidden_dim, 
            num_classes, 
            dropout_rate=0.0, 
            attn_dropout_rate=0.0, 
            pe_type="learned"
    ):
        super().__init__(num_channels, 16, num_classes, embedding_dim)
        assert (embedding_dim % num_heads == 0) and (img_dim % patch_dim == 0)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.seq_length = int((img_dim // patch_dim) ** 3)

        if pe_type == "learned":
            self.pe = LearnedPositionalEncoding(
                self.embedding_dim, self.seq_length
            )
        elif pe_type == "fixed":
            self.pe = FixedPositionalEncoding(
                self.embedding_dim, self.seq_length
            )

        self.pe_drop = nn.Dropout(self.dropout_rate)

        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_dim=hidden_dim,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
        )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.linear_proj = nn.Sequential(
            nn.BatchNorm3d(128), 
            nn.ReLU(inplace=True), 
            nn.Conv3d(128, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        x1, x2, x3, x = self.encode(x)
        x = self.linear_proj(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (B, H, W, D, C)
        x = x.view(x.size(0), -1, self.embedding_dim) # (B, H × W × D, C)
        
        x = self.pe(x)
        x = self.pe_drop(x)
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        x = self._reshape_output(x)

        y = self.decode(
            x1, x2, x3, x
        )

        return y
    

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (batch_size, embedding_dim, H, W, D)
        return x


def get_default(_pe_type="learned"):
    img_dim = 128
    num_classes = 4
    num_channels = 4
    patch_dim = 8
    model = TransBTS(
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        num_classes=num_classes,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        pe_type=_pe_type,
    )

    return model


if __name__ == '__main__':
    with torch.no_grad():
        import os
        device = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=device)
        model = TransBTS(_pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
