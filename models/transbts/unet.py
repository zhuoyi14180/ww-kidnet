"""
This script is adapted from the following repository: 
        https://github.com/MIC-DKFZ/BraTS2017

Author: Zhuoyi Zhang
Date: 2024-06-12
"""

import torch.nn as nn
import torch
from models.transbts.blocks import InitConv, EnBlock, EnDown, DeBlock, DeUp
from models.modules import ResBlock


class UNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4, embedding_dim=512, final_act=nn.Softmax):
        super().__init__()

        # (1, 16, 128, 128, 128)
        self.en_block1 = nn.Sequential(
            InitConv(in_channels=in_channels, out_channels=base_channels, dropout_rate=0.2), 
            EnBlock(in_channels=base_channels)
        )

        # (1, 32, 64, 64, 64)
        self.en_block2 = nn.Sequential(
            EnDown(in_channels=base_channels, out_channels=base_channels*2), 
            EnBlock(in_channels=base_channels*2), 
            EnBlock(in_channels=base_channels*2)
        )

        # (1, 64, 32, 32, 32)
        self.en_block3 = nn.Sequential(
            EnDown(in_channels=base_channels*2, out_channels=base_channels*4), 
            EnBlock(in_channels=base_channels * 4), 
            EnBlock(in_channels=base_channels * 4)
        )

        # (1, 128, 16, 16, 16)
        self.en_block4 = nn.Sequential(
            EnDown(in_channels=base_channels*4, out_channels=base_channels*8), 
            EnBlock(in_channels=base_channels * 8), 
            EnBlock(in_channels=base_channels * 8), 
            EnBlock(in_channels=base_channels * 8), 
            EnBlock(in_channels=base_channels * 8)
        )


        self.Softmax = nn.Softmax(dim=1)

        self.en_block5 = nn.Sequential(
            DeBlock(in_channels=embedding_dim, hidden_channels=embedding_dim//4, out_channels=embedding_dim//4), 
            ResBlock(DeBlock(in_channels=embedding_dim//4))
        )

        self.de_up3 = DeUp(in_channels=embedding_dim//4, out_channels=embedding_dim//8)
        self.de_block3 = ResBlock(DeBlock(in_channels=embedding_dim//8))

        self.de_up2 = DeUp(in_channels=embedding_dim//8, out_channels=embedding_dim//16)
        self.de_block2 = ResBlock(DeBlock(in_channels=embedding_dim//16))

        self.de_up1 = DeUp(in_channels=embedding_dim//16, out_channels=embedding_dim//32)
        self.de_block1 = ResBlock(DeBlock(in_channels=embedding_dim//32))

        self.end = nn.Conv3d(embedding_dim // 32, num_classes, kernel_size=1)

        if final_act is not None:
            self.final_act = final_act(dim=1)
        else:
            self.final_act = None


    def encode(self, x):
        x1 = self.en_block1(x)
        x2 = self.en_block2(x1)
        x3 = self.en_block3(x2)
        y = self.en_block4(x3)
        return x1, x2, x3, y # (1, 128, 16, 16, 16)
    
    
    def decode(self, x1, x2, x3, x):
        x4 = self.en_block5(x)
        y3 = self.de_up3(x4, x3)
        y3 = self.de_block3(y3)
        y2 = self.de_up2(y3, x2)
        y2 = self.de_block2(y2)
        y1 = self.de_up1(y2, x1)
        y1 = self.de_block1(y1)
        y = self.end(y1)
        return y if self.final_act is None else self.final_act(y)

    def forward(self, x):
        return x


if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=device)
        model = UNet(in_channels=4, base_channels=16, num_classes=4)
        model.cuda()
        output = model(x)
        print('output: ', output.shape)
