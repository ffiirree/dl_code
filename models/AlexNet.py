import torch
import torch.nn as nn

__all__ = ['AlexNet']


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 224 x 224 x 3
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 55 x 55 x 96
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 27 x 27 x 96
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 27 x 27 x 256
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 13 x 13 x 256
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 13 x 13 x 256
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 13 x 13 x 384
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 13 x 13 x 256
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 6 x 6 x 256
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),

            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x),
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x