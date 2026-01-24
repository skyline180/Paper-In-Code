import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class ResNetBackbone(ResNet):
    """
    - Supports ResNet-18/34/50/101
    """

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        out_dim=2048,
    ):
        super().__init__(
            block,
            layers,
            num_classes=1000,  # dummy, removed later
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

        # Remove classifier
        self.fc = nn.Identity()
        self.out_dim = out_dim

    def forward(self, x):
        # Copied from torchvision with fc removed
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet18(**kwargs):
    return ResNetBackbone(BasicBlock, [2, 2, 2, 2], out_dim=512, **kwargs)


def resnet34(**kwargs):
    return ResNetBackbone(BasicBlock, [3, 4, 6, 3], out_dim=512, **kwargs)


def resnet50(**kwargs):
    return ResNetBackbone(Bottleneck, [3, 4, 6, 3], out_dim=2048, **kwargs)


def resnet101(**kwargs):
    return ResNetBackbone(Bottleneck, [3, 4, 23, 3], out_dim=2048, **kwargs)
