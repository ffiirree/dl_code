import argparse
import os
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

__all__ = ['DenosingAutoencoder']

# Same as Autoencoder
class DenosingAutoencoder(nn.Module):
    def __init__(self, image_size, nz):
        super().__init__()
        
        self.image_size = image_size
        self.nz = nz
        
        self.encoder = nn.Sequential(
            nn.Linear(self.image_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.nz),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.image_size ** 2),
            nn.Tanh()
        )

    def forward(self, x):
        size = x.size()
        x = torch.flatten(x, 1)
        z = self.encoder(x)
        x = self.decoder(z)
        return x.reshape(size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--nz', type=int, default=3, help='size of the latent z vector')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam, default=0.9')
    parser.add_argument('--output-dir', default='logs/dae')

    opt = parser.parse_args()
    print(opt)

    dataset = torchvision.datasets.MNIST(
        root=os.path.expanduser('~/data/datasets'),
        download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    denosing_ae = DenosingAutoencoder(image_size=28, nz=opt.nz).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(denosing_ae.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    for epoch in range(200):
        for idx, (images, labels) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc='epoch #{:>3d}'.format(epoch),
            leave=False):

            images = images.to(device)
            noise = torch.randn_like(images)
            noised_images = images + noise

            denosing_ae.zero_grad()
            output = denosing_ae(noised_images)
            loss = criterion(output, images)
            loss.backward()

            optimizer.step()

            if idx % 100 == 0:
                torchvision.utils.save_image(noised_images.detach(), './{}/noised_images.png'.format(opt.output_dir, idx), normalize=True)
                torchvision.utils.save_image(output.detach(), './{}/fake.png'.format(opt.output_dir, idx), normalize=True)
