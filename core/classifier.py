import torch

from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from torchvision import transforms

from tqdm import tqdm

from dataset.imagenet_dataset import ImagenetDataset
from core.resnet import ResNet
from core.image import ImageDataGenerator
from core.lib import loadData


class Classifier(object):
    """
    Classifier Wrapper to Train Data
    """

    def __init__(self):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device %s" % self.device)

        # train_dataset = ImagenetDataset(train=True, transform=train_transform)
        # self.classes = train_dataset.class_names
        # self.trainloader = \
        # DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        # test_dataset = ImagenetDataset(train=True, transform=test_transform)
        # self.testloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2)
        dataset = loadData("dataset/imagenet.h5")
        train_imagegen = ImageDataGenerator(
            rescale=1./255, 
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_imagegen = ImageDataGenerator(
            rescale=1./255
        )
        train_x = dataset["train"]["data"]
        train_y = dataset["train"]["target"]
        self.train_len = len(train_x)
        self.trainloader = train_imagegen.flow(train_x, train_y,batch_size=32,shuffle=True)
        test_x = dataset["val"]["data"]
        test_y = dataset["val"]["target"]
        self.test_len = len(test_x)
        self.testloader = test_imagegen.flow(test_x, test_y,batch_size=32,shuffle=True)
        self.classes = dataset["class_index"]
        
        self.net = ResNet(len(self.classes))
        self.net.to(self.device)
        if self.device == 'cuda':
            self.net = nn.DataParallel(self.net)
            cudnn.benchmark = True

    def train(self, epoch):
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        correct = 0
        total = 0
        acc = 0
        barformat = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt} {postfix[loss]}[{remaining} {rate_fmt} accuracy:{postfix[accuracy]}%]"
        pbar = tqdm(total=self.train_len, bar_format=barformat, postfix={"accuracy":acc, "loss":0, 5:0})
        for batch_idx, (data, target) in enumerate(self.trainloader):
            pbar.set_description("epoch {0}".format(epoch+1))
            data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.int64, device=self.device)
            optimizer.zero_grad()
            outputs = self.net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            acc = 100.0*correct/total
            pbar.postfix["accuracy"] = "{0:.2f}".format(acc)
            pbar.postfix["loss"] = "{0:10.3f}".format(train_loss)
            if self.device == "cuda":
                if batch_idx % 50 == 0:
                    pbar.update(50)
            else:
                pbar.update(1)
            if batch_idx >= self.train_len:
                break
            # print(batch_idx, train_loss)

        print("\nEpoch %d complete... " % epoch)

    def test(self, epoch):
        self.net.eval()
        correct = 0
        total = 0
        acc = 0
        with torch.no_grad():
            barformat = "{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{remaining} {rate_fmt} {postfix}%]"
            pbar = tqdm(total=self.test_len, bar_format=barformat, postfix={"accuracy":acc})
            for _, (data, target) in enumerate(self.testloader):
                pbar.set_description("epoch {0}".format(epoch+1))
                data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.int64, device=self.device)
                outputs = self.net(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                acc = 100.*correct/total
                pbar.set_postfix({"accuracy":acc})
                if self.device == "cuda":
                    if batch_idx % 50 == 0:
                        pbar.update(50)
                else:
                    pbar.update(1)
        print("Saving Model...")
        state = {
            'net': self.net.state_dict(),
            'epoch': epoch,
            'accuracy': acc
        }
        torch.save(state, 'checkpoint/checkpoint.t7')
