import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(dim, norm='gn'):
    if norm == 'bn':
        res = nn.BatchNorm3d(dim)
    elif norm == 'gn':
        res = nn.GroupNorm(8, dim)
    elif norm == 'in':
        res = nn.InstanceNorm3d(dim)
    else:
        raise ValueError('Invalid normalization type {}, check first.'.format(norm))
    return res


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        x = self.conv(x)
        return F.dropout3d(x, self.dropout)


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super().__init__()

        self.norm1 = normalize(in_channels, norm=norm)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.norm2 = normalize(in_channels, norm=norm)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.relu(x1)
        x1 = self.conv1(x1)
        y = self.norm2(x1)
        y = self.relu(y)
        y = self.conv2(y)
        return y + x


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=16, num_classes=4):
        super().__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=base_channels, dropout=0.2)
        self.EnBlock1 = EnBlock(in_channels=base_channels)
        self.EnDown1 = EnDown(in_channels=base_channels, out_channels=base_channels * 2)

        self.EnBlock2_1 = EnBlock(in_channels=base_channels * 2)
        self.EnBlock2_2 = EnBlock(in_channels=base_channels * 2)
        self.EnDown2 = EnDown(in_channels=base_channels*2, out_channels=base_channels * 4)

        self.EnBlock3_1 = EnBlock(in_channels=base_channels * 4)
        self.EnBlock3_2 = EnBlock(in_channels=base_channels * 4)
        self.EnDown3 = EnDown(in_channels=base_channels*4, out_channels=base_channels * 8)

        self.EnBlock4_1 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_2 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_3 = EnBlock(in_channels=base_channels * 8)
        self.EnBlock4_4 = EnBlock(in_channels=base_channels * 8)

    def forward(self, x): # (batch_size, channels, height, width, depth)
        x = self.InitConv(x) # (1, 16, 128, 128, 128)

        x1_1 = self.EnBlock1(x)
        x1_2 = self.EnDown1(x1_1) # (1, 32, 64, 64, 64)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_2 = self.EnDown2(x2_1) # (1, 64, 32, 32, 32)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_2 = self.EnDown3(x3_1) # (1, 128, 16, 16, 16)

        out = self.EnBlock4_1(x3_2)
        out = self.EnBlock4_2(out)
        out = self.EnBlock4_3(out)
        out = self.EnBlock4_4(out) # (1, 128, 16, 16, 16)

        return x1_1, x2_1, x3_1, out
