![Python](https://img.shields.io/badge/Python-3.x-blue)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20TensorFlow%20%7C%20Keras%20%7C%20Scikit--Learn-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)
![GitHub Repo Size](https://img.shields.io/github/repo-size/skyline180/Paper-In-Code)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=skyline180.Paper-In-Code)


---

# Paper-In-Code

# Paper in Code 📄➡️💻

**Paper in Code** is a collection of clean, well-documented implementations of influential deep learning research papers.  
The goal of this repository is to bridge the gap between theory and practice by converting research papers into readable, reproducible code.

This repo is intended for:
- Learning how research papers are implemented
- Experimenting with architectures and ideas
- Building intuition by reading and modifying code
- Serving as a reference for future projects

---

## 📚 Implemented / Planned Papers

### Computer Vision
- **ResNet** – *Deep Residual Learning for Image Recognition* (He et al., 2015)
- **EfficientNet** – *Rethinking Model Scaling for Convolutional Neural Networks* (Tan & Le, 2019)

### Model Compression & Optimization
- **Knowledge Distillation** – *Distilling the Knowledge in a Neural Network* (Hinton et al., 2015)

> More papers will be added progressively (Vision Transformers, MobileNet, ConvNeXt, etc.)

---

## 📁 Repository Structure
```
paper-in-code/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── models/
│ ├── resnet/
│ │ ├── resnet.py
│ │ └── README.md
│ │
│ ├── efficientnet/
│ │ ├── efficientnet.py
│ │ └── README.md
│ │
│ └── ...
│
├── distillation/
│ ├── knowledge_distillation.py
│ └── README.md
│
├── experiments/
│ ├── resnet_cifar10.py
│ ├── efficientnet_imagenet.py
│ └── notebooks/
│
└── utils/
├── data_loader.py
├── train.py
└── metrics.py
```

Each paper/model folder contains:
- Original paper reference
- Model architecture
- Training & evaluation code
- Notes and explanations

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/paper-in-code.git
cd paper-in-code

pip install -r requirements.txt
```

