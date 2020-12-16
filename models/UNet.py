import torch
import torch.nn as nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, channels):
        super(UNet, self).__init__()

        self.downblock1 = nn.Sequential(
            DoubleConv2d(channels, 64)
        )

        self.downblock2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(64, 128)
        )

        self.downblock3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(128, 256)
        )

        self.downblock4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(256, 512)
        )

        self.u = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)
        )

        self.upblock1 = nn.Sequential(
            DoubleConv2d(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, stride=2)
        )

        self.upblock2 = nn.Sequential(
            DoubleConv2d(512, 256),
            nn.ConvTranspose2d(256, 128, 2, stride=2)
        )

        self.upblock3 = nn.Sequential(
            DoubleConv2d(256, 128),
            nn.ConvTranspose2d(128, 64, 2, stride=2)
        )

        self.upblock4 = nn.Sequential(
            DoubleConv2d(128, 64),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, x):
        down1 = self.downblock1(x)
        down2 = self.downblock2(down1)
        down3 = self.downblock3(down2)
        down4 = self.downblock4(down3)

        u = self.u(down4)

        up1 = self.upblock1(torch.cat([down4, u], 1))
        up2 = self.upblock2(torch.cat([down3, up1], 1))
        up3 = self.upblock3(torch.cat([down2, up2], 1))
        out = self.upblock4(torch.cat([down1, up3], 1))
        return out
        

if __name__ == '__main__':
    print(UNet(3))

