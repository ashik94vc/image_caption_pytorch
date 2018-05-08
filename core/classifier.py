import sys
import torch

from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
from dataset.imagenet_dataset import ImagenetDataset
from core.resnet import ResNet
from core.lib import saveModel
sys.path.append('../')


class Classifier(object):
    """
    Classifier Wrapper to Train Data
    """

    def __init__(self):
        train_transform = transforms.Compose({
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        })
        test_transform = transforms.Compose({
            transforms.ToTensor()
        })

        self.device = 'cuda' if torch.cuda_is_available() else 'cpu'

        train_dataset = ImagenetDataset(train=True, transform=train_transform)
        self.classes = train_dataset.class_names
        self.trainloader = \
        DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_dataset = ImagenetDataset(train=True, transform=test_transform)
        self.testloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)

        self.net = ResNet(len(self.classes))
        self.net.to(self.device)
        if self.device == 'cuda':
            self.net = nn.DataParallel(self.net)
            cudnn.benchmark = True

    def train(self, epoch):
        print('\nEpoch %d ' % epoch)
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        acc = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            outputs = self.net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += target.size(0)
            train_loss += loss.item()
            acc = 100.*correct/total
            print(batch_idx, train_loss)

        print("\nEpoch %d complete with accuracy %f " % (epoch, acc))
        print("Saving Model")
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
            'accuracy': acc
        }
        torch.save(state, '../checkpoint/checkpoint.t7')

