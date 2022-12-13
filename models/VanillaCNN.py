import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.max_pooling(out)
        return out


class CNN(nn.Module):
    def __init__(self, in_channel, classes_num):
        super(CNN, self).__init__()

        self.conv1 = ConvBlock(in_channel, 4, 5, 2, 2)
        self.conv2 = ConvBlock(4, 8, 3, 1, 1)
        self.conv3 = ConvBlock(8, 16, 3, 1, 1)
        self.conv4 = ConvBlock(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(128, 256)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, classes_num)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.flatten(out, 1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.fc2(out)
        return out
