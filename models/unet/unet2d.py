from torch import nn
import torch
from models.unet.blocks import DoubleConv2d, Down2d, Up2d


class UNet2D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_channels=64, bilinear=False, final_act=nn.Softmax):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv2d(n_channels, base_channels)
        self.down1 = Down2d(base_channels, base_channels * 2)
        self.down2 = Down2d(base_channels * 2, base_channels * 4)
        self.down3 = Down2d(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.bottleneck = Down2d(base_channels * 8, (base_channels * 16) // factor)
        self.up1 = Up2d((base_channels * 16), (base_channels * 8) // factor, bilinear)
        self.up2 = Up2d(base_channels * 8, (base_channels * 4) // factor, bilinear)
        self.up3 = Up2d(base_channels * 4, (base_channels * 2) // factor, bilinear)
        self.up4 = Up2d((base_channels * 2), base_channels, bilinear)
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        if final_act is not None:
            self.final_act = final_act(dim=1)
        else:
            self.final_act = None

    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.bottleneck(x3)
        x3 = self.up1(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up3(x2, x1)
        x = self.up4(x1, x)
        logits = self.outc(x)
        return logits if self.final_act is None else self.final_act(logits)
    

def get_default():
    return UNet2D()
