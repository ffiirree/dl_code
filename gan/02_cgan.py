import argparse
import os
from numpy.lib.ufunclike import fix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, default=0.5')
parser.add_argument('--output-dir', default='cgan_logs')

opt = parser.parse_args()
print(opt)

dataset = torchvision.datasets.MNIST(
    train=True,
    download=True,
    root=os.path.expanduser('~/data/datasets/'),
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.embeds = nn.Embedding(10, 10)
        
        self.net = nn.Sequential(
            nn.Linear(opt.nz + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh() # Training failed when using the 'sigmoid'
        )
    
    def forward(self, z, labels):
        x = torch.cat((z, self.embeds(labels)), 1)
        return self.net(x)
    
class Discriminatar(nn.Module):
    def __init__(self):
        super(Discriminatar, self).__init__()
        
        # if this is the same as G, it's would be hard to convergance
        self.embeds = nn.Embedding(10, 200)
        
        self.net = nn.Sequential(
            nn.Linear(784 + 200, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels):
        return self.net(torch.cat((x, self.embeds(labels)), 1))
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminatar = Discriminatar().to(device)

criterion = nn.BCELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(discriminatar.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

fixed_z = torch.randn(opt.batch_size, opt.nz).to(device)
fixed_labels = torch.randint(0, 10, [opt.batch_size]).to(device)
print(fixed_labels)
for epoch in range(200):
    for idx, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc='epoch #{:>3d}'.format(epoch), leave=False):
        ##############################################################################
        # real data
        batch_size = images.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        real_data, real_labels = images.to(device), labels.to(device)
        
        discriminatar.zero_grad()
        
        output = discriminatar(real_data.flatten(1), real_labels)
        d_real_loss = criterion(output, real)
        d_real_loss.backward()
        
        # fake data
        z = torch.randn(batch_size, opt.nz).to(device)
        fake_labels = torch.randint(0, 10, [batch_size]).to(device)
        fake_data = generator(z, fake_labels)
        
        output = discriminatar(fake_data.detach(), fake_labels)
        d_fake_loss = criterion(output, fake)
        d_fake_loss.backward()
        
        optimizer_d.step()
        
        ##############################################################################
        generator.zero_grad()
        output = discriminatar(fake_data, fake_labels)
        g_loss = criterion(output, real)
        g_loss.backward()
        optimizer_g.step()
        
        if idx % 100 == 0:
            gen_images = generator(fixed_z, fixed_labels)
            torchvision.utils.save_image(gen_images.detach().reshape(batch_size, 1, 28, 28), './{}/fake_fixed.png'.format(opt.output_dir), normalize=True)
            torchvision.utils.save_image(fake_data.detach().reshape(batch_size, 1, 28, 28), './{}/fake.png'.format(opt.output_dir), normalize=True)