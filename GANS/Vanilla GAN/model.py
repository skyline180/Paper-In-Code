
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generates synthetic MNIST images from random noise.
    Input:  (batch_size, latent_dim)
    Output: (batch_size, 1, 28, 28)
    """

    def __init__(self, latent_dim=100, image_dim=784):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, noise):
        image = self.network(noise)
        return image.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """
    Classifies an MNIST image as real or generated.
    Input:  (batch_size, 1, 28, 28)
    Output: (batch_size,) raw logits
    """

    def __init__(self, image_dim=784):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, image):
        image = image.view(image.size(0), -1)
        return self.network(image).view(-1)

