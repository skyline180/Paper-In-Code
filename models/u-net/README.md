# U-Net (PyTorch)

This directory contains a clean PyTorch implementation of the U-Net for image segmentation tasks.

# U-Net consists of:

- Encoder (Downsampling path): extracts features using convolution + pooling
- Decoder (Upsampling path): reconstructs spatial resolution using transposed convolutions
- Skip connections: concatenate encoder features with decoder features to preserve fine details

## Usage
```python
from models.unet.unet import UNet

model = UNet(in_channels=3, num_classes=1)
```

## Output Shape

For an input image:

```[B, C, H, W]```

The model outputs:

```[B, num_classes, H, W]```

## Loss Functions
```python
import torch.nn as nn
```
```python
# Binary segmentation
criterion = nn.BCEWithLogitsLoss()
```
```python
# Multi-class segmentation
criterion = nn.CrossEntropyLoss()
```

## Reference

Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).

https://arxiv.org/abs/1505.04597
