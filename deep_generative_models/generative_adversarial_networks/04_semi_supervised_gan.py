import argparse
import os
from PIL.Image import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='mnist | cifar10 | imagenet')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--image-size', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of the generator filters')
parser.add_argument('--ndf', type=int, default=64, help='size of the discriminator filters')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--output-dir', default='logs/sgan')

opt = parser.parse_args()
print(opt)

dataset = torchvision.datasets.MNIST(
    train=True,
    download=True,
    root=os.path.expanduser('~/data/datasets/'),
    transform=transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

num_channels = 1
num_classes = 10

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # input : (batch_size, nz, 1, 1)
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 8, 4, 4)
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 4, 8, 8)
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size: (batch_size, ngf * 2, 16, 16)
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size : (batch_size, ngf, 32, 32)
            nn.ConvTranspose2d(opt.ngf, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size : (batch_size, num_channels, 64, 64)
        )

    def forward(self, z):
        return self.net(z)
    
class Discriminatar(nn.Module):
    def __init__(self):
        super(Discriminatar, self).__init__()
        
        self.net = nn.Sequential(
            # input size : (batch_size, num_channels, 64, 64)
            nn.Conv2d(num_channels, opt.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, opt.ndf, 32, 32)
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #state size : (batch_size, opt.ndf * 2, 16, 16)
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, opt.ndf * 4, 8, 8)
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch_size, opt.ndf * 8, 4, 4)
        )
        
        feature_map_size = opt.image_size // 2 ** 4
        
        self.adv_layer = nn.Sequential(
            nn.Linear(opt.ndf * 8 * feature_map_size ** 2, 1),
            nn.Sigmoid()
            # state size : (batch_size, 1)
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(opt.ndf * 8 * feature_map_size ** 2, num_classes + 1),
            nn.Softmax(dim=1)
            # state size : (batch_size, num_classes + 1)
        )
        
    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        
        adv_out = self.adv_layer(x)
        aux_out = self.aux_layer(x)
        return adv_out, aux_out
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminatar = Discriminatar().to(device)

adv_loss = nn.BCELoss()
aux_loss = nn.CrossEntropyLoss()

optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(discriminatar.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

for epoch in range(200):
    for idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc='epoch #{:>3d}'.format(epoch), leave=False):
        ##############################################################################
        # real data
        batch_size = images.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        real_data, real_labels = images.to(device), labels.to(device)
        
        z = torch.randn(batch_size, opt.nz, 1, 1).to(device)
        fake_data = generator(z)
        
        discriminatar.zero_grad()
        
        adv_out, aux_out = discriminatar(real_data)
        d_real_loss = (adv_loss(adv_out, real) + aux_loss(aux_out, real_labels)) / 2
        d_real_loss.backward()
        
        # fake data
        adv_out, aux_out = discriminatar(fake_data.detach())
        d_fake_loss = adv_loss(adv_out, fake)
        d_fake_loss.backward()
        
        optimizer_d.step()
        
        ##############################################################################
        generator.zero_grad()
        adv_out, aux_out = discriminatar(fake_data)
        g_loss = adv_loss(adv_out, real)
        g_loss.backward()
        optimizer_g.step()
        
        if idx % 100 == 0:
            torchvision.utils.save_image(fake_data.detach(), './{}/fake.png'.format(opt.output_dir, idx), normalize=True)