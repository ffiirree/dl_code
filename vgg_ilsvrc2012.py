from logging import log
from utils.utils import torch_models
from torchvision import transforms
from torchvision.datasets import imagenet
from torchvision.transforms.transforms import CenterCrop, Resize
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from utils import make_logger
import argparse


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, device, loader, epoch, optimizer, criterion):
    model.train()

    for idx, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)

        # logger.info(output.isnan().any())

        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output=output, target=targets, topk=(1, 5))
        logger.info(
            f'[epoch {epoch:>3}/{args.epochs}@{idx + 1:>5}/{len(loader)}] loss={loss.item():.4f}, acc@1={acc1.item():.3f}, acc@5={acc5.item():.3f}')


def validate(model, device, loader, criterion):
    model.eval()

    acc1 = 0
    acc5 = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)

            out = model(images)
            loss = criterion(out, targets)

            acc1_, acc5_ = accuracy(output=out, target=targets, topk=(1, 5))
            acc1 += acc1_
            acc5 += acc5_

            top1 = acc1 / (((idx + 1)))
            top5 = acc5 / (((idx + 1)))
            logger.info(
                f'{idx + 1:>5}/{len(loader)}: * Acc@1 {top1.item():.3f} Acc@5 {top5.item():.3f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',              type=str,
                        default='resnet18', choices=torch_models())
    parser.add_argument('--dataset',            type=str,
                        default='~/data/datasets/ILSVRC2012/')
    parser.add_argument('--pretrained',         type=bool,  default=False)
    parser.add_argument('-j', '--workers',      type=int,   default=32)
    parser.add_argument('--epochs',             type=int,   default=90)
    parser.add_argument('-b', '--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight-decay',       type=float, default=1e-4)
    parser.add_argument('--output-dir',         type=str,   default='logs')
    return parser.parse_args()


if __name__ == "__main__":
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')

    args = parse_args()
    logger = make_logger('imagenet12')
    logger.info(args)

    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model = nn.DataParallel(model)
    model.to(device)
    logger.info(f'use gpus: {model.device_ids}')

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(
            args.dataset,
            'train',
            transform=T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet(
            args.dataset,
            'val',
            transform=T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train(model, device, train_loader, epoch=epoch,
              optimizer=optimizer, criterion=criterion)
        validate(model, device, val_loader, criterion=criterion)
