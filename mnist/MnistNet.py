import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MnistNet']

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(0.05)
        )
    
    def forward(self, x):
        return self.conv(x)

# All ReLU: 99.61% VS All sigmoid: 99.20%
# BN + Dropout: 99.66%
# 1. 完全使用Sigmoid，相比完全使用ReLU，训练慢且准确率低
class MnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistNet, self).__init__()

        self.features = nn.Sequential(
            ConvBlock( 1, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 64, 3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

        # self.features.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x