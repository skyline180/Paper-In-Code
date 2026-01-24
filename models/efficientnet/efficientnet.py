import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# Utility functions

def round_filters(filters, width_mult, divisor=8):
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_mult):
    return int(math.ceil(depth_mult * repeats))


# Basic blocks

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        squeezed = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, in_ch, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        expand_ratio,
        stride,
        kernel_size,
        se_ratio=0.25,
        drop_rate=0.0,
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        hidden_ch = in_ch * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.SiLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(
                hidden_ch,
                hidden_ch,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_ch,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            SqueezeExcite(hidden_ch, se_ratio),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]

        self.block = nn.Sequential(*layers)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                out = F.dropout(out, p=self.drop_rate, training=True)
            out = out + x
        return out


# EfficientNet

class EfficientNet(nn.Module):
    def __init__(
        self,
        width_mult,
        depth_mult,
        num_classes=1000,
        dropout=0.2,
    ):
        super().__init__()

        base_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, round_filters(base_channels, width_mult), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(round_filters(base_channels, width_mult)),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        in_ch = round_filters(base_channels, width_mult)

        config = [
            # expand, out, repeats, stride, kernel
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        total_blocks = sum(round_repeats(c[2], depth_mult) for c in config)
        block_id = 0

        for expand, out_ch, repeats, stride, k in config:
            out_ch = round_filters(out_ch, width_mult)
            repeats = round_repeats(repeats, depth_mult)

            for i in range(repeats):
                s = stride if i == 0 else 1
                drop = 0.2 * block_id / total_blocks
                self.blocks.append(
                    MBConv(in_ch, out_ch, expand, s, k, drop_rate=drop)
                )
                in_ch = out_ch
                block_id += 1

        head_ch = round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Model variants

def efficientnet_b0(num_classes=1000):
    return EfficientNet(1.0, 1.0, num_classes)


def efficientnet_b1(num_classes=1000):
    return EfficientNet(1.0, 1.1, num_classes)


def efficientnet_b2(num_classes=1000):
    return EfficientNet(1.1, 1.2, num_classes)


def efficientnet_b3(num_classes=1000):
    return EfficientNet(1.2, 1.4, num_classes)


def efficientnet_b4(num_classes=1000):
    return EfficientNet(1.4, 1.8, num_classes)


def efficientnet_b5(num_classes=1000):
    return EfficientNet(1.6, 2.2, num_classes)


def efficientnet_b6(num_classes=1000):
    return EfficientNet(1.8, 2.6, num_classes)


def efficientnet_b7(num_classes=1000):
    return EfficientNet(2.0, 3.1, num_classes)

