import argparse
import time
import torch
import torch.nn as nn
from torch.serialization import load
import torchvision
import torchvision.transforms as T
from utils import make_logger
from tqdm import tqdm

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
        
        output = model(images)
        loss = criterion(output, targets)
        
        # logger.info(output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc1, acc5 = accuracy(output=output, target=targets, topk=(1, 5))
        
        if (idx+1) % 30 == 0:
            logger.info(f'[epoch {epoch:>3}/{args.epochs}@{idx + 1:>5}/{len(loader)}] lr={optimizer.param_groups[0]["lr"]}, loss={loss.item():.4f}, acc@1={acc1.item():.3f}, acc@5={acc5.item():.3f}')

        
def test(model, device, loader, criterion):
    model.eval()
    
    acc1= 0
    acc5 = 0
    
    with tqdm(total=len(loader), dynamic_ncols=True, desc='test', unit='batch', leave=False) as pbar:

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                out = model(images)
                loss = criterion(out, targets)
            
            acc1_, acc5_ = accuracy(output=out, target=targets, topk=(1, 5))
            acc1 += acc1_
            acc5 += acc5_
            
            pbar.update()
    return (acc1 / len(loader)).item(), (acc5 / len(loader)).item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',            type=str,   default='~/data/datasets/CIFAR100/')
    parser.add_argument('--pretrained',         type=bool,  default=False)
    parser.add_argument('-j', '--workers',      type=int,   default=16)
    parser.add_argument('--epochs',             type=int,   default=120)
    parser.add_argument('-b', '--batch_size',   type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.05)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight-decay',       type=float, default=1e-4)
    parser.add_argument('--output-dir',         type=str,   default='logs')
    parser.add_argument('--patience',           type=int,   default=6)
    return parser.parse_args()

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')
    
    args = parse_args()
    logger = make_logger('cifar100')
    logger.info(args)
    
    model = torchvision.models.vgg11_bn(pretrained=args.pretrained)
    model = nn.DataParallel(model)
    model.to(device)
    logger.info(f'use gpus: {model.device_ids}')
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            args.dataset,
            train=True,
            download=False,
            transform=T.Compose([
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.5074,0.4867,0.4411), std=(0.2011,0.1987,0.2025))
            ])
        ),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            args.dataset,
            train=False,
            download=False,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize(mean=(0.5074,0.4867,0.4411), std=(0.2011,0.1987,0.2025))
            ])
        ),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=True
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.patience)
    
    for epoch in range(args.epochs):
        train(model, device, train_loader, epoch, optimizer, criterion)
        acc1, acc5 = test(model, device, test_loader, criterion)
        logger.info(f'[test] acc@1={acc1}, acc@5={acc5}')
        scheduler.step(acc1)
        
    model_filename = f'{args.output_dir}/vgg11_bn__{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.pth'
    torch.save(model.module.state_dict(), model_filename)
    logger.info(f'Model saved: {model_filename}!!')
    
    