# Knowledge Distillation

This module implements knowledge distillation using PyTorch.

## Features
- Teacher-student training
- KL divergence + cross-entropy loss
- Configurable temperature and alpha

## Usage

```python
from knowledge_distillation import train_distillation

train_distillation(student, teacher, dataloader, device)
```

## Reference

https://arxiv.org/pdf/1503.02531
