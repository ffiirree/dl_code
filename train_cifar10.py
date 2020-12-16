from numpy.core.fromnumeric import ptp
from models import Cifar10Net
import os.path as osp
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def train(model, data_loader, optimizer, criterion, scheduler, epoch):
    train_loss = 0

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, train_loss / 2000))
            scheduler.step(train_loss / 200)
            train_loss = 0.0


def test(model, data_loader, epoch):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('\tTest Epoch #{:>2}: {}/{} ({:>3.2f}%)'.format(epoch, correct, total, 100. * correct / total))


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(osp.expanduser('~/data/datasets'), train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    test_data = torchvision.datasets.CIFAR10(osp.expanduser('~/data/datasets'), train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Cifar10Net()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=1)

    for epoch in range(75):
        train(model, train_loader, optimizer, criterion, scheduler, epoch)
        test(model, test_loader, epoch)

