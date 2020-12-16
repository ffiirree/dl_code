import argparse
import os
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
parser.add_argument('--output-dir', default='vanila_gan_logs')

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
        
        self.net = nn.Sequential(
            nn.Linear(opt.nz, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh() # Training failed when using the 'sigmoid'
        )
    
    def forward(self, x):
        return self.net(x)
    
class Discriminatar(nn.Module):
    def __init__(self):
        super(Discriminatar, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
        
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
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        real_data = data[0].to(device)
        
        discriminatar.zero_grad()
        
        output = discriminatar(real_data.flatten(1))
        d_real_loss = criterion(output, real_labels)
        d_real_loss.backward()
        
        # fake data
        noise = torch.randn(batch_size, opt.nz).to(device) # latent variable: z
        fake_data = generator(noise)
        
        output = discriminatar(fake_data.detach())
        d_fake_loss = criterion(output, fake_labels)
        d_fake_loss.backward()
        
        optimizer_d.step()
        
        ##############################################################################
        generator.zero_grad()
        output = discriminatar(fake_data)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_g.step()
        
        if idx % 100 == 0:
            torchvision.utils.save_image(fake_data.detach().reshape(batch_size, 1, 28, 28), './{}/fake_{}.png'.format(opt.output_dir, idx), normalize=True)