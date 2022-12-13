import torch
from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, conv_num):
        super(ConvBlock, self).__init__()

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                )
            ]
            + [
                nn.Sequential(
                    nn.Conv2d(in_channels=out_channel,
                              out_channels=out_channel,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                ) for _ in range(conv_num - 1)
            ]
        )

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.max_pooling(x)
        return x


class VggNet(nn.Module):
    def __init__(self, in_channel, classes_num):
        super(VggNet, self).__init__()

        self.conv1 = ConvBlock(in_channel, 64, 2)
        self.conv2 = ConvBlock(64, 128, 2)
        self.conv3 = ConvBlock(128, 256, 3)
        self.conv4 = ConvBlock(256, 512, 3)
        self.conv5 = ConvBlock(512, 512, 3)

        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, classes_num),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
