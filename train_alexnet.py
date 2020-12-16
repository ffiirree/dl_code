from models import AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = AlexNet()
print(net)