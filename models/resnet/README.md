# ResNet (PyTorch)

This directory contains a clean PyTorch implementation of ResNet:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

## Usage

```python
from models.resnet.resnet import resnet50

model = resnet50(num_classes=10)
```

## Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
Deep Residual Learning for Image Recognition.  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

https://arxiv.org/abs/1512.03385
