import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

__all__ = ['LinearAutoencoder']

class LinearAutoencoder(nn.Module):
    def __init__(self, image_size, nz):
        super().__init__()
        
        self.image_size = image_size
        self.nz = nz
        
        self.encoder = nn.Sequential(
            nn.Linear(image_size ** 2, 512, bias=False),
            nn.Linear(512, self.nz),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.nz, 512, bias=False),
            nn.Linear(512, image_size ** 2, bias=False)
        )

    def forward(self, x):
        size = x.size()
        x = torch.flatten(x, 1)
        z = self.encoder(x)
        x = self.decoder(z)
        return x.reshape(size)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
    parser.add_argument('--output-dir', default='logs/linear_ae')

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

    linear_autoencoder = LinearAutoencoder(image_size=28, nz=opt.nz).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(linear_autoencoder.parameters(), lr=opt.lr, momentum=0.9)

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
