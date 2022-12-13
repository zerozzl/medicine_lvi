import torch
import torch.nn as nn


class BaicsBlock(nn.Module):

    def __init__(self, channel, downsample=False, first=False):
        super(BaicsBlock, self).__init__()
        self.downsample = downsample

        if downsample:
            self.convd = nn.Conv2d(in_channels=int(channel / 2),
                                   out_channels=channel,
                                   kernel_size=1,
                                   stride=2,
                                   padding=0,
                                   bias=False)

            self.conv1 = nn.Conv2d(in_channels=int(channel / 2),
                                   out_channels=channel,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channel)
        else:
            self.conv1 = nn.Conv2d(in_channels=channel,
                                   out_channels=channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = nn.Conv2d(in_channels=channel,
                               out_channels=channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.convd(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, channel, downsample=False, first=False):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.first = first

        if downsample:
            self.convd = nn.Conv2d(in_channels=channel * 2,
                                   out_channels=channel * 4,
                                   kernel_size=1,
                                   stride=2,
                                   padding=0,
                                   bias=False)

            self.conv1 = nn.Conv2d(in_channels=channel * 2,
                                   out_channels=channel,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(channel)

            self.conv2 = nn.Conv2d(in_channels=channel,
                                   out_channels=channel,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(channel)
        else:
            if first:
                self.convd = nn.Conv2d(in_channels=channel,
                                       out_channels=channel * 4,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

                self.conv1 = nn.Conv2d(in_channels=channel,
                                       out_channels=channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
                self.bn1 = nn.BatchNorm2d(channel)
            else:
                self.conv1 = nn.Conv2d(in_channels=channel * 4,
                                       out_channels=channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
                self.bn1 = nn.BatchNorm2d(channel)

            self.conv2 = nn.Conv2d(in_channels=channel,
                                   out_channels=channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = nn.Conv2d(in_channels=channel,
                               out_channels=channel * 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(channel * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample or self.first:
            identity = self.convd(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, in_channel, classes_num):
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channel,
                               out_channels=32,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self.make_layer(block, 64, blocks_num[0])
        self.layer_2 = self.make_layer(block, 128, blocks_num[1], downsample=True)
        self.layer_3 = self.make_layer(block, 256, blocks_num[2], downsample=True)
        self.layer_4 = self.make_layer(block, 512, blocks_num[3], downsample=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        if block == BaicsBlock:
            self.fc = nn.Linear(512, classes_num)
        elif block == Bottleneck:
            self.fc = nn.Linear(2048, classes_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, block, channel, block_num, downsample=False):
        layers = [block(channel, downsample, first=not downsample)]
        for _ in range(1, block_num):
            layers.append(block(channel, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18(in_channel, classes_num):
    return ResNet(BaicsBlock, [2, 2, 2, 2], in_channel, classes_num=classes_num)


def ResNet34(in_channel, classes_num):
    return ResNet(BaicsBlock, [3, 4, 6, 3], in_channel, classes_num=classes_num)


def ResNet50(in_channel, classes_num):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channel, classes_num=classes_num)


def ResNet101(in_channel, classes_num):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_channel, classes_num=classes_num)


def ResNet152(in_channel, classes_num):
    return ResNet(Bottleneck, [3, 8, 36, 3], in_channel, classes_num=classes_num)
