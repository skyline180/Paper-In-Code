import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()

        inter_channels = bn_size * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv1x1(in_channels, inter_channels)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = conv3x3(inter_channels, growth_rate)

        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        super().__init__()

        self.layers = nn.ModuleList()
        channels = in_channels

        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)
            channels += growth_rate

        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()

        out_channels = int(in_channels * compression)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = conv1x1(in_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config: List[int] = [6, 12, 24, 16],
        num_classes=1000,
        bn_size=4,
        compression=0.5,
        drop_rate=0.0,
    ):
        super().__init__()

        num_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks + transitions
        self.features = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.features.append(block)
            num_channels = block.out_channels

            if i != len(block_config) - 1:
                trans = Transition(num_channels, compression=compression)
                self.features.append(trans)
                num_channels = trans.out_channels

        # Final layers
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))

        for layer in self.features:
            x = layer(x)

        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# Variants
def densenet121(num_classes=1000, **kwargs):
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        num_classes=num_classes,
        **kwargs
    )


def densenet169(num_classes=1000, **kwargs):
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 32, 32],
        num_classes=num_classes,
        **kwargs
    )


def densenet201(num_classes=1000, **kwargs):
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 48, 32],
        num_classes=num_classes,
        **kwargs
    )
