
# Vanilla GAN

A PyTorch implementation of the original Generative Adversarial
Network (GAN), trained on the MNIST handwritten-digit dataset.

## 1. Research Paper

**Title:** Generative Adversarial Nets

**Authors:** Ian J. Goodfellow and collaborators

**Published:** 2014, NeurIPS

**Paper:** https://arxiv.org/abs/1406.2661

**Original conference paper:**
https://papers.nips.cc/paper/5423-generative-adversarial-nets

## 2. Overview

A Generative Adversarial Network (GAN) consists of two neural networks:

- Generator (G): Creates synthetic samples from random noise.
- Discriminator (D): Distinguishes real samples from generated samples.

The two networks compete during training. The generator learns to
produce increasingly realistic samples, while the discriminator
learns to identify generated samples.

## 3. Architecture

### Generator

Input:
- Random latent vector z of dimension 100.

Architecture:
- Linear(100, 256) + LeakyReLU
- Linear(256, 512) + BatchNorm + LeakyReLU
- Linear(512, 1024) + BatchNorm + LeakyReLU
- Linear(1024, 784) + Tanh

Output:
- Grayscale image of shape 1 x 28 x 28.
- Pixel values in the range [-1, 1].

### Discriminator

Input:
- Grayscale image of shape 1 x 28 x 28.

Architecture:
- Linear(784, 1024) + LeakyReLU + Dropout
- Linear(1024, 512) + LeakyReLU + Dropout
- Linear(512, 256) + LeakyReLU
- Linear(256, 1)

Output:
- One raw logit per image.
- BCEWithLogitsLoss is used for classification.

## 4. Objective Function

The original GAN objective is:

min_G max_D V(D, G) =

E[x ~ p_data(x)] [log D(x)]
+ E[z ~ p_z(z)] [log(1 - D(G(z)))]

where:

- x is a real sample.
- z is a random latent vector.
- G(z) is a generated sample.
- D(x) estimates whether a sample is real.

In this implementation, the discriminator and generator are
optimized in alternating steps using binary cross-entropy losses.

The generator uses the non-saturating objective:

L_G = -E[z] [log D(G(z))]

This encourages generated samples to be classified as real.

## 5. Dataset

MNIST handwritten digits:
- 60,000 training images.
- 28 x 28 grayscale images.
- 10 digit classes, from 0 to 9.

The GAN does not use the digit labels during training.

Dataset is downloaded automatically by torchvision.

## 6. Project Structure

Vanilla GAN/
├── README.md
├── model.py
├── train.py
├── generate.py
├── requirements.txt
├── data/
├── checkpoints/
└── outputs/

The data, checkpoints, and outputs directories are created
automatically when needed.

## 7. Installation

Install dependencies:

    pip install -r requirements.txt

## 8. Training

Run:

    python train.py

The default configuration uses:
- Latent dimension: 100
- Batch size: 128
- Epochs: 50
- Learning rate: 0.0002
- Optimizer: Adam
- Dataset: MNIST

The script saves the generator checkpoint to:

    checkpoints/vanilla_gan.pt

Sample images are saved in outputs/ after selected epochs.

## 9. Generate Images

Generate 64 images:

    python generate.py

Generate 100 images:

    python generate.py --num-images 100

Specify an output path:

    python generate.py --num-images 64 --output outputs/my_samples.png

Use a fixed random seed:

    python generate.py --num-images 64 --seed 42

## 10. Important Notes

- This is a fully connected Vanilla GAN, not a DCGAN.
- The generator and discriminator are trained without digit labels.
- GAN training can be unstable and may suffer from mode collapse.
- Loss values alone do not reliably measure image quality.
- The generated images are expected to improve over training,
  but high quality is not guaranteed.

## 11. Future Improvements

- Experiment with different latent dimensions.
- Compare optimizers and learning rates.
- Plot generator and discriminator losses.
- Compare Vanilla GAN against DCGAN.
- Evaluate generated image quality.

## License

This implementation is provided for educational and research use.
Check the original paper and dataset terms before redistribution.
