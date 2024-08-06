import torch
from torch import nn
from models.unet.blocks import DoubleConv3d, Down3d, Up3d


class UNet3D(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_channels=64, final_act=nn.Softmax):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3d(n_channels, base_channels, mid_channels=base_channels // 2)
        self.down1 = Down3d(base_channels, base_channels * 2, mid_channels=base_channels)
        self.down2 = Down3d(base_channels * 2, base_channels * 4, mid_channels=base_channels * 2)
        self.down3 = Down3d(base_channels * 4, base_channels * 8, mid_channels=base_channels * 4)
        self.up1 = Up3d(base_channels * 8, base_channels * 4)
        self.up2 = Up3d(base_channels * 4, base_channels * 2)
        self.up3 = Up3d(base_channels * 2, base_channels)
        self.outc = nn.Conv3d(base_channels, n_classes, kernel_size=1)

        if final_act is not None:
            self.final_act = final_act(dim=1)
        else:
            self.final_act = None

    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x2 = self.up1(x3, x2)
        x1 = self.up2(x2, x1)
        x = self.up3(x1, x)
        logits = self.outc(x)
        return logits if self.final_act is None else self.final_act(logits)
    

def get_default():
    return UNet3D()


if __name__ == "__main__":
    with torch.no_grad():
        import os
        device = torch.device('cuda:0')
        x = torch.rand((2, 4, 128, 128, 128), device=device)
        model = UNet3D(4, 4)
        model.cuda()
        y = model(x)
        print(y.shape)