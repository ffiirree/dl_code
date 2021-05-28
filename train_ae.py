import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models import AutoEncoder, UNet
from utils import make_logger
from utils import *

class VOCDatasetTransforms(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    CROP_SIZE = 384
    IGNORE_LABEL = 255
    
    def __init__(self):
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

def train(model, device, loader, optimizer, criterion, epoch, args):
    model.train()
    
    for idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        score = model(images)
        loss = criterion(score, masks)
        
        # print(score[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torchvision.utils.save_image(images[0], f'logs/voc/{idx}_image.png', normalize=True)
        torchvision.utils.save_image(masks[0].float(), f'logs/voc/{idx}_mask.png', normalize=True)
        torchvision.utils.save_image(score.max(1)[1][0].float(), f'logs/voc/{idx}_pred.png', normalize=True)
        
        logger.info(f'[epoch {epoch:>3}/{args.epochs} - {idx:>3}/{len(loader)}] loss = {loss.item()}')
    
def validate(model, device, loader):
    model.eval()
    
    for idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            score = model(images)
            
        logger.info(mean_acc(score.max(1)[1], masks, 21, ignore_index=255))
        logger.info(mean_iou(score.max(1)[1], masks, 21, ignore_index=255))
        
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=0.1)
    parser.add_argument('--workers',    type=int,   default=16)
    parser.add_argument('--batch-size', type=int,   default=32)
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
    model.to(device)
    logger.info(f'use gps: {model.device_ids}')
    
    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.SBDataset(
    #         os.path.expanduser('~/data/datasets/VOC'), image_set='train_noval', mode='segmentation', download=False,
    #         transforms=VOCDatasetTransforms()
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.VOCSegmentation(
            os.path.expanduser('~/data/datasets/VOC'), year='2012', image_set='train', download=False,
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
            os.path.expanduser('~/data/datasets/VOC'), year='2012', image_set='val', download=False,
            transforms=VOCDatasetTransforms()
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logger.info(f'Loaded {len(val_loader.dataset)} images')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, criterion, epoch=epoch, args=args)
        validate(model, device, val_loader)
        
    torch.save(model.module.state_dict(), 'ae_voc.pth')
    logger.info('Done!')