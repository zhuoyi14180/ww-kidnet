import torch.nn as nn
import torch.nn.functional as F
from models.utils.consts import NormType
import torch


class InitConv(nn.Module):
    def __init__(self, in_channels=4, out_channels=16, dropout_rate=0.2):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout_rate = dropout_rate


    def forward(self, x):
        x = self.conv(x)
        x = F.dropout3d(x, self.dropout_rate)
        return x


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm=NormType.GROUP_NORM):
        super().__init__()

        self.net = nn.Sequential(
            norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        y = self.net(x)
        return y + x


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    

class DeBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()

        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1), 
            nn.BatchNorm3d(hidden_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm3d(hidden_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    

class DeUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x = self.conv1(x)
        y = self.conv2(x)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


