import torch
from torch import nn
from torch.nn import LeakyReLU, Tanh

class ResNet(nn.Module):

    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.num_classes = num_classes

        ''' 64x64 images with 3 channels as input. '''
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3) #Output: 64x32x32
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1) #Output: 64x16x16
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  #Output: 64x16x16
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) #Output: 128x8x8
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1) #Output: 128x8x8
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1) #Output: 256x4x4
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1) #Output: 256x4x4
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1) #Output: 512x4x4
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1) #Output: 512x4x4
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.avgpool = nn.AvgPool2d(2, 2) #Output: 512x2x2
        self.linear1 = nn.Linear(2048, 2048)
        self.linear2 = nn.Linear(2048, num_classes)

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.log_smax = nn.Softmax(1)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn64(x)
        x = self.pool1(x)
        x = self.relu(x)
        residue = x
        x = self.conv2(x)
        x = self.bn64(x)
        x = self.conv2(x)
        x = self.bn64(x)
        x += residue
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn128(x)
        x = self.relu(x)
        residue = x
        x = self.conv4(x)
        x = self.bn128(x)
        x = self.conv4(x)
        x = self.bn128(x)
        x += residue
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn256(x)
        x = self.relu(x)
        residue = x
        x = self.conv6(x)
        x = self.bn256(x)
        x = self.conv6(x)
        x = self.bn256(x)
        x += residue
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn512(x)
        x = self.relu(x)
        residue = x
        x = self.conv8(x)
        x = self.bn512(x)
        x = self.conv8(x)
        x = self.bn512(x)
        x += residue
        # x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.log_smax(x)
        return x
