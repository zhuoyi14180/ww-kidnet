import torch
from torch import nn
from .block import DoubleConv3d, Down3d, Up3d


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3d(n_channels, 64, mid_channels=32)
        self.down1 = Down3d(64, 128, mid_channels=64)
        self.down2 = Down3d(128, 256, mid_channels=128)
        self.down3 = Down3d(256, 512, mid_channels=256)
        self.up1 = Up3d(512, 256)
        self.up2 = Up3d(256, 128)
        self.up3 = Up3d(128, 64)
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        y2 = self.up1(x3, x2)
        y1 = self.up2(y2, x1)
        y = self.up3(y1, x)
        y = self.outc(y)
        return self.softmax(y)
    


if __name__ == "__main__":
    with torch.no_grad():
        import os
        device = torch.device('cuda:0')
        x = torch.rand((2, 4, 128, 128, 128), device=device)
        model = UNet3D(4, 4)
        model.cuda()
        y = model(x)
        print(y.shape)