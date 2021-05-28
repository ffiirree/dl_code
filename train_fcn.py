import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from models import FCN8s, FCN16s, FCN32s
from utils import make_logger

class VOCDatasetTransforms(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

    def __call__(self, input, target):
        input = self.transform(input)

        target = torch.from_numpy(np.array(target, dtype=np.int32)).long()
        target[target == 255] = 0

        return input, target

    def __repr__(self):
        body = [self.__class__.__name__]
        return '\n'.join(body)

# def test(model):
#     pass

def train(model, device, train_loader, optimizer, criterion):

    for epoch in range(50):

        train_loss = 0

        for batch_index, (image, target) in enumerate(train_loader):

            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            score = model(image)

            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_index % 20 == 0 and batch_index != 0:
                logger.info('epoch #{:>2} [{:>4}/{}]: {}'.format(epoch, batch_index, len(train_loader), train_loss / 20))
                train_loss = 0
                torchvision.utils.save_image(score.max(1)[1].cpu().float(), f'{args.output_dir}/{batch_index}_p.png', nrow=4, normalize=True)
                torchvision.utils.save_image(target.float(), f'{args.output_dir}/{batch_index}_t.png', nrow=4, normalize=True)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA IS NOT AVAILABLE!!"
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str,   default='fcn8s')
    parser.add_argument('--workers',    type=int,   default=16)
    parser.add_argument('--batch-size', type=int,   default=1)
    parser.add_argument('--output-dir', type=str,   default='logs')

    args = parser.parse_args()

    logger = make_logger(args.model, args.output_dir)
    logger.info(args)

    if args.model == 'fcn8s':
        model = FCN8s(num_classes=21)
    elif args.model == 'fcn16s':
        model = FCN16s(num_classes=21)
    else:
        model = FCN32s(num_classes=21)

    model = nn.DataParallel(model)
    model.to(device)
    logger.info(f'use gpus: {model.device_ids}')
    
    train_dataset = torchvision.datasets.VOCSegmentation(
            os.path.expanduser('~/data/datasets/VOC'), image_set='train', download=False,
            transforms=VOCDatasetTransforms()
        )
    
    print(len(train_dataset))


    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.VOCSegmentation(
    #         os.path.expanduser('~/data/datasets/VOC'), year='2012', image_set='val', download=False,
    #         transforms=VOCDatasetTransforms()
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True
    # )

    # optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-9, momentum=0.99, weight_decay=0.0005)

    # criterion = torch.nn.CrossEntropyLoss()

    # train(model, device, train_loader, optimizer, criterion)
