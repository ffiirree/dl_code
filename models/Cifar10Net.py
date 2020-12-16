import torch
import torch.nn as nn

__all__ = ['Cifar10Net']


# 86.13%
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)

class Cifar10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32 x 32 x 64
            ResidualUnit(64, 64, stride=1),
            ResidualUnit(64, 64, stride=1),

            ResidualUnit(64, 128, stride=1),
            ResidualUnit(128, 128, stride=1),
            ResidualUnit(128, 128, stride=2),

            # 16 x 16 x 128
            ResidualUnit(128, 256, stride=1),
            ResidualUnit(256, 256, stride=2),

            # 8 x 8 x 256
            ResidualUnit(256, 512, stride=1),
            ResidualUnit(512, 512, stride=2),
            # 4 x 4 x 512
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x