import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from models import FCN32s
from models import FCN16s
from models import FCN8s

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
    if torch.cuda.device_count() > 1:
        print('{} GPUs'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        
    model.to(device)
    
    if not os.path.exists('./fcn_out'):
        os.makedirs('./fcn_out')
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
                print('epoch #{:>2} [{:>4}/{}]: {}'.format(epoch, batch_index, len(train_loader), train_loss / 20))
                train_loss = 0
                torchvision.utils.save_image(score.max(1)[1].cpu().float(), 'fcn_out/{}_p.png'.format(batch_index), nrow=4, normalize=True)
                torchvision.utils.save_image(target.float(), 'fcn_out/{}_t.png'.format(batch_index), nrow=4, normalize=True)


if __name__ == '__main__':
    model = FCN8s(num_classes=21)
    
    vgg16 = torchvision.models.vgg16_bn(pretrained=True)
    pretrained_dict = vgg16.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SBDataset(
            os.path.expanduser('~/data/datasets/VOC'), image_set='train_noval', mode='segmentation', download=False,
            transforms=VOCDatasetTransforms()
        ),
        batch_size=1, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.VOCSegmentation(
            os.path.expanduser('~/data/datasets/VOC'), year='2012', image_set='val', download=False,
            transforms=VOCDatasetTransforms()
        ),
        batch_size=1, shuffle=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-9, momentum=0.99, weight_decay=0.0005)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    train(model, device, train_loader, optimizer, criterion)
