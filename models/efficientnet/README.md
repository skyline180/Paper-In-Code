# EfficientNet (PyTorch)

PyTorch implementation of EfficientNet (B0–B7) using MBConv blocks
and compound scaling.

## Usage

```python
from models.efficientnet.efficientnet import efficientnet_b0

model = efficientnet_b0(num_classes=10)

