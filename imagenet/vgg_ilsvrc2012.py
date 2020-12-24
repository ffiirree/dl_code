from torchvision import transforms
from torchvision.datasets import imagenet
from torchvision.transforms.transforms import CenterCrop, Resize
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T


def validate(model, device, loader, criterion):
    model.eval()

    acc1= 0
    acc5 = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            model.to(device)
            images, targets = images.to(device), targets.to(device)

            
            out = model(images)
            loss = criterion(out, targets)

            _, predicted = out.topk(5, 1, True, True)
            predicted = predicted.t()

            correct = predicted.eq(targets.view(1, -1).expand_as(predicted))

            acc1 += correct[:1].reshape(-1).float().sum(0, keepdim=True)
            acc5 += correct[:5].reshape(-1).float().sum(0, keepdim=True)

            top1 = acc1 / (((idx + 1) * targets.size(0)))
            top5 = acc5 / (((idx + 1) * targets.size(0)))
            print('{:>5}/{}: * Acc@1 {:.3f} Acc@5 {:.3f}'.format(idx + 1, len(loader), top1.item() * 100.0, top5.item() * 100.0))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.vgg16_bn(pretrained=True)

    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageNet('F:/ImageNet_ILSVRC2012/', 'val', download=False, transform=transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8, 
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    validate(model, device, loader, criterion)
        

