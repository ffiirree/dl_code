import os
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


class AutoEncoder(nn.Module):
    def __init__(self, len):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 * 1, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, len),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(len, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 28 * 28 * 1),
            nn.Tanh()
        )

    def forward(self, x):
        size = x.size()
        x = torch.flatten(x, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x.reshape(size)



if __name__=='__main__':
    os.makedirs("ae", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, ], [0.5])
            ])
        ),
        batch_size=16,
        shuffle=True,
    )

    net = AutoEncoder(3)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        net.train()
        train_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 200 == 0 and i != 0:
                print('Train Epoch # {} [{:>5}/{}]\tloss: {:.6f}'.format(epoch, i * len(images), len(dataloader.dataset), train_loss / 200))
                # scheduler.step(train_loss / 200)
                train_loss = 0

                save_image(output, 'ae/{}.png'.format(i), nrow=4, normalize=True)

