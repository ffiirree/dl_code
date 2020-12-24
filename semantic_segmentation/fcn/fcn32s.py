from numpy.lib import npyio
import torch
import torch.nn as nn
from torch.nn.modules import adaptive

__all__ = ['FCN32s']

class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()

        self.conv1_1    = nn.Conv2d(3, 64, 3, padding=100)
        self.bn1_1      = nn.BatchNorm2d(64)
        self.relu1_1    = nn.ReLU(inplace=True)
        self.conv1_2    = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2      = nn.BatchNorm2d(64)
        self.relu1_2    = nn.ReLU(inplace=True)
        self.pool1      = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1    = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1      = nn.BatchNorm2d(128)
        self.relu2_1    = nn.ReLU(inplace=True)
        self.conv2_2    = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2      = nn.BatchNorm2d(128)
        self.relu2_2    = nn.ReLU(inplace=True)
        self.pool2      = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1    = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1      = nn.BatchNorm2d(256)
        self.relu3_1    = nn.ReLU(inplace=True)
        self.conv3_2    = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2      = nn.BatchNorm2d(256)
        self.relu3_2    = nn.ReLU(inplace=True)
        self.conv3_3    = nn.Conv2d(256, 256, 3, padding=True)
        self.bn3_3      = nn.BatchNorm2d(256)
        self.relu3_3    = nn.ReLU(inplace=True)
        self.pool3      = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1    = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1      = nn.BatchNorm2d(512)
        self.relu4_1    = nn.ReLU(inplace=True)
        self.conv4_2    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2      = nn.BatchNorm2d(512)
        self.relu4_2    = nn.ReLU(inplace=True)
        self.conv4_3    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3      = nn.BatchNorm2d(512)
        self.relu4_3    = nn.ReLU(inplace=True)
        self.pool4      = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1      = nn.BatchNorm2d(512)
        self.relu5_1    = nn.ReLU(inplace=True)
        self.conv5_2    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2      = nn.BatchNorm2d(512)
        self.relu5_2    = nn.ReLU(inplace=True)
        self.conv5_3    = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3      = nn.BatchNorm2d(512)
        self.relu5_3    = nn.ReLU(inplace=True)
        self.pool5      = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.fc6        = nn.Conv2d(512, 4096, 7)
        self.bn6        = nn.BatchNorm2d(4096)
        self.relu6      = nn.ReLU(inplace=True)
        self.drop6      = nn.Dropout2d()

        self.fc7        = nn.Conv2d(4096, 4096, 1)
        self.bn7        = nn.BatchNorm2d(4096)
        self.relu7      = nn.ReLU(inplace=True)
        self.drop7      = nn.Dropout2d()

        self.score_fr   = nn.Conv2d(4096, num_classes, 1)

        self.upscore    = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, bias=False)

    def forward(self, x):
        out = x

        out = self.relu1_1(self.bn1_1(self.conv1_1(out)))
        out = self.relu1_2(self.bn1_2(self.conv1_2(out)))
        out = self.pool1(out)

        out = self.relu2_1(self.bn2_1(self.conv2_1(out)))
        out = self.relu2_2(self.bn2_2(self.conv2_2(out)))
        out = self.pool2(out)

        out = self.relu3_1(self.bn3_1(self.conv3_1(out)))
        out = self.relu3_2(self.bn3_2(self.conv3_2(out)))
        out = self.relu3_3(self.bn3_3(self.conv3_3(out)))
        out = self.pool3(out)

        out = self.relu4_1(self.bn4_1(self.conv4_1(out)))
        out = self.relu4_2(self.bn4_2(self.conv4_2(out)))
        out = self.relu4_3(self.bn4_3(self.conv4_3(out)))
        out = self.pool4(out)

        out = self.relu5_1(self.bn5_1(self.conv5_1(out)))
        out = self.relu5_2(self.bn5_2(self.conv5_2(out)))
        out = self.relu5_3(self.bn5_3(self.conv5_3(out)))
        out = self.pool5(out)

        out = self.relu6(self.bn6(self.fc6(out)))
        out = self.drop6(out)

        out = self.relu7(self.bn7(self.fc7(out)))
        out = self.drop7(out)

        out = self.score_fr(out)
        out = self.upscore(out)

        return out[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())