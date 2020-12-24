from models import MnistNet
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms


def train(train_loader, model, device, criterion, optimizer, scheduler, epoch):
    net.train()
    train_loss = 0
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % 200 == 0 and i != 0:
            print('Train Epoch # {} [{:>5}/{}]\tloss: {:.6f}'.format(epoch, i * len(data), len(train_loader.dataset),
                                                                     train_loss / 200))
            scheduler.step(train_loss / 200)
            train_loss = 0


def test(test_loader, model, device, epoch):
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            _, predicted = torch.max(output.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('\tTest Epoch #{:>2}: {}/{} ({:>3.2f}%)'.format(epoch, correct, total, 100. * correct / total))


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_data = torchvision.datasets.MNIST(
        train=True,
        download=True,
        root='./data',
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_data = torchvision.datasets.MNIST(
        train=False,
        download=True,
        root='./data',
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = MnistNet()
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(75):
        train(train_loader, net, device, criterion, optimizer, scheduler, epoch)
        test(test_loader, net, device, epoch)
