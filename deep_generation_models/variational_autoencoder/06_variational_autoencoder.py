import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam, default=0.9')
parser.add_argument('--output-dir', default='vae_out')

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

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_fc1 = nn.Sequential(
            nn.Linear(image_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder_fc2 = nn.Sequential(
            nn.Linear(512, opt.nz)
        )

        self.encoder_fc3 = nn.Sequential(
            nn.Linear(512, opt.nz)
        )

        self.decoder = nn.Sequential(
            nn.Linear(opt.nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, image_size ** 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        size = x.size()

        x = torch.flatten(x, 1)

        mu = self.encoder_fc2(self.encoder_fc1(x))
        logvar = self.encoder_fc3(self.encoder_fc1(x))

        std = torch.exp(0.5* logvar)
        eps = torch.randn_like(logvar)

        z = mu + eps * std

        x = self.decoder(z)
        return x.reshape(size), mu, logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Reconstruction + KL divergence losses summed over all elements and batch
def criterion(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

for epoch in range(200):
    for idx, (images, labels) in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc='epoch #{:>3d}'.format(epoch),
        leave=False):

        images = images.to(device)

        vae.zero_grad()
        output, mu, logvar = vae(images)
        loss = criterion(output, images, mu, logvar)
        loss.backward()

        optimizer.step()

        if idx % 100 == 0:
            torchvision.utils.save_image(output.detach(), './{}/fake.png'.format(opt.output_dir, idx))
