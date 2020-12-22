import argparse
import os
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--nz', type=int, default=3, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam, default=0.9')
parser.add_argument('--output-dir', default='ae_out')

opt = parser.parse_args()
print(opt)

image_size = 28

dataset = torchvision.datasets.MNIST(
    root=os.path.expanduser('~/data/datasets'),
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5, ))
    ])
)

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(image_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, opt.nz),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(opt.nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, image_size ** 2),
            nn.Tanh()
        )

    def forward(self, x):
        size = x.size()
        x = torch.flatten(x, 1)
        z = self.encoder(x)
        x = self.decoder(z)
        return x.reshape(size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

linear_autoencoder = LinearAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_autoencoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer = optim.SGD(linear_autoencoder.parameters(), lr=opt.lr, momentum=0.9) # not work

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

for epoch in range(200):
    for idx, (images, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc='epoch #{:>3d}'.format(epoch),
        leave=False):

        images = images.to(device)

        linear_autoencoder.zero_grad()
        output = linear_autoencoder(images)
        loss = criterion(output, images)
        loss.backward()

        optimizer.step()

        if idx % 100 == 0:
            torchvision.utils.save_image(output.detach(), './{}/fake.png'.format(opt.output_dir, idx), normalize=True)
