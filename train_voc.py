import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models import AutoEncoder, UNet, FCN8s
from utils import make_logger
from utils import *

class_name = (
    'background', 'aeroplane', 'bicycle',
    'bird', 'boat', 'bottle',
    'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
)

class VOCDatasetTransforms(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    CROP_SIZE = 384
    IGNORE_LABEL = 255
    
    def __init__(self) -> None:
        super().__init__()
        self.transform = T.Compose([
            T.RandomCrop(self.CROP_SIZE, pad_if_needed=True, fill=0, padding_mode='constant'),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD)
        ])
        
        self.target_transform = T.Compose([
            T.RandomCrop(self.CROP_SIZE, pad_if_needed=True, fill=self.IGNORE_LABEL, padding_mode='constant'),
            T.RandomHorizontalFlip(),
        ])

    def __call__(self, input, target):
        seed = np.random.randint(2147483647)
        # torchvision >= v0.8.0
        torch.manual_seed(seed)
        input = self.transform(input)
        
        torch.manual_seed(seed)
        target = self.target_transform(target)
        
        target = TF.to_tensor(np.asarray(target, np.int32)).long()
        # print(target.shape)

        return input, target[0].long()

    def __repr__(self):
        body = [self.__class__.__name__]
        return '\n'.join(body)

class VOCValDatasetTransforms(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    CROP_SIZE = 384
    IGNORE_LABEL = 255

    def __init__(self) -> None:
        super().__init__()
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD)
        ])

    def __call__(self, input, target):
        input = self.transform(input)    
        target = TF.to_tensor(np.asarray(target, np.int32)).long()

        return input, target[0].long()

    def __repr__(self):
        body = [self.__class__.__name__]
        return '\n'.join(body)

def train(model, device, loader, optimizer, criterion, epoch, args):
    model.train()
    
    for idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        score = model(images)
        loss = criterion(score, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = score.argmax(dim=1)

        masks[masks==255] = 0

        if (idx + 1) % 10 == 0:        
            torchvision.utils.save_image(images[0], f'logs/voc/{idx}_image.png', normalize=True)
            torchvision.utils.save_image(masks[0].float(), f'logs/voc/{idx}_mask.png', normalize=True)
            torchvision.utils.save_image(pred[0].float(), f'logs/voc/{idx}_pred.png', normalize=True)
        
        logger.info(f'[epoch {epoch:>3}/{args.epochs} - {idx:>3}/{len(loader)}] loss = {loss.item()}, {pred[0].cpu().unique().numpy()}/{masks[0].cpu().unique().numpy()}')
    
def validate(model, device, loader):
    model.eval()

    acc = torch.zeros([21]).to(device)
    iou = torch.zeros([21]).to(device)

    with tqdm(total=len(loader), dynamic_ncols=True, desc='volidation', unit='batch', leave=False) as pbar:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                score = model(images)
                
            pred = score.argmax(dim=1)
            
            acc += mean_acc(pred, masks, 21, ignore_index=255)
            iou += mean_iou(pred, masks, 21, ignore_index=255)
            pbar.update()

        for name, acc, iou in zip(class_name, acc, iou):
            logger.info(f'{name:^15}\t{(acc * 100.) / len(loader):<5.3f}\t{(iou * 100.) / len(loader):5.3f}')
            
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--lr',         type=float, default=0.01)
    parser.add_argument('--workers',    type=int,   default=8)
    parser.add_argument('--batch-size', type=int,   default=3)
    parser.add_argument('--output-dir', type=str,   default='logs')
    return parser.parse_args()

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA IS NOT AVAILABLE!!'
    device = torch.device('cuda')
    
    args = parse_args()
    logger = make_logger('train_autoencoder', log_dir=args.output_dir)
    logger.info(args)
    
    model = UNet(n_channels=3, n_classes=21)

    model = nn.DataParallel(model)
    model.module.load_state_dict(torch.load('unet_voc.pth'))
    model.to(device)
    logger.info(f'use gps: {model.device_ids}')
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SBDataset(
            'G:\PASCAL_VOC', image_set='train_noval', mode='segmentation', download=False,
            transforms=VOCDatasetTransforms()
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    logger.info(f'Loaded {len(train_loader.dataset)} images')

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.VOCSegmentation(
            'G:\PASCAL_VOC', year='2012', image_set='val', download=False,
            transforms=VOCDatasetTransforms()
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logger.info(f'Loaded {len(val_loader.dataset)} images')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.00005)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, criterion, epoch=epoch, args=args)
        validate(model, device, val_loader)

    torch.save(model.module.state_dict(), 'unet_voc.pth')
    logger.info('Done!')