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
parser.add_argument('--output-dir', default='logs/dcgan')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

if opt.dataset == 'mnist':
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
    nc = 1
elif opt.dataset == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(
        train=True,
        download=True,
        root=os.path.expanduser('~/data/datasets'),
        transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    nc = 3
elif opt.dataset == 'imagenet':
    dataset = torchvision.datasets.ImageNet(
        root=os.path.expanduser('~/data/datasets/ImageNet_ILSVRC2012'),
        split='train',
        download=False,
        transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    nc = 3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = torchvision.datasets.LSUN(
        root=os.path.expanduser('~/data/datasets'),
        classes=classes,
        transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    nc=3
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # input : (batch_size, nz, 1, 1)
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 8, 4, 4)
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size : (batch_size, ngf * 4, 8, 8)
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size: (batch_size, ngf * 2, 16, 16)
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size : (batch_size, ngf, 32, 32)
            nn.ConvTranspose2d(opt.ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size : (batch_size, nc, 64, 64)
        )

    def forward(self, z):
        return self.net(z)
    
class Discriminatar(nn.Module):
    def __init__(self):
        super(Discriminatar, self).__init__()
        
        self.net = nn.Sequential(
            # input size : (batch_size, nc, 64, 64)
            nn.Conv2d(nc, opt.ndf, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size : (batch_size, 1, 1, 1)
        )
        
    def forward(self, x):
        return self.net(x).flatten()
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminatar = Discriminatar().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(discriminatar.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

for epoch in range(200):
    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc='epoch #{:>3d}'.format(epoch), leave=False):
        ##############################################################################
        # real data
        batch_size = data[0].size(0)
        real = torch.ones(batch_size).to(device)
        fake = torch.zeros(batch_size).to(device)
        
        real_data = data[0].to(device)
        
        discriminatar.zero_grad()
        
        output = discriminatar(real_data)
        d_real_loss = criterion(output, real)
        d_real_loss.backward()
        
        # fake data
        z = torch.randn(batch_size, opt.nz, 1, 1).to(device)
        fake_data = generator(z)
        
        output = discriminatar(fake_data.detach())
        d_fake_loss = criterion(output, fake)
        d_fake_loss.backward()
        
        optimizer_d.step()
        
        ##############################################################################
        generator.zero_grad()
        output = discriminatar(fake_data)
        g_loss = criterion(output, real)
        g_loss.backward()
        optimizer_g.step()
        
        if idx % 100 == 0:
            torchvision.utils.save_image(fake_data.detach(), './{}/fake_{}.png'.format(opt.output_dir, idx), normalize=True)