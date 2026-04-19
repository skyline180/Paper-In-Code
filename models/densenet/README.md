# DenseNet (PyTorch)

This directory contains a clean PyTorch implementation of DenseNet:

- DenseNet-121
- DenseNet-169
- DenseNet-201

## Usage
```
from models.densenet.densenet import densenet121

model = densenet121(num_classes=10)
```

## Overview

DenseNet (Densely Connected Convolutional Networks) connects each layer to every other layer in a feed-forward fashion.
Instead of summing features like ResNet, DenseNet **concatenates** them, which improves feature reuse and gradient flow.

## Reference

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
Densely Connected Convolutional Networks.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

https://arxiv.org/abs/1608.06993
