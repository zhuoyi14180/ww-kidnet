from enum import Enum, auto
from torch import nn


class WeightInit(Enum):
    kaiming = auto()
    xavier = auto()


class NormType(Enum):
    GROUP_NORM = 'gn'
    BATCH_NORM_1D = "bn1"
    INSTANCE_NORM_1D = 'in1'
    BATCH_NORM_2D = "bn2"
    INSTANCE_NORM_2D = 'in2'
    BATCH_NORM_3D = 'bn3'
    INSTANCE_NORM_3D = 'in3'
    
    
    def __call__(self, num_channels, num_groups=None):
        if self == NormType.GROUP_NORM:
            num_groups = num_groups or 8
            return nn.GroupNorm(num_groups, num_channels)
        elif self == NormType.BATCH_NORM_3D:
            return nn.BatchNorm3d(num_channels)
        elif self == NormType.INSTANCE_NORM_3D:
            return nn.InstanceNorm3d(num_channels)
        elif self == NormType.BATCH_NORM_2D:
            return nn.BatchNorm2d(num_channels)
        elif self == NormType.INSTANCE_NORM_2D:
            return nn.InstanceNorm2d(num_channels)
        elif self == NormType.BATCH_NORM_1D:
            return nn.BatchNorm1d(num_channels)
        elif self == NormType.INSTANCE_NORM_1D:
            return nn.InstanceNorm1d(num_channels)

    