
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Generator, Discriminator


# -----------------------
# Configuration
# -----------------------
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5

DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")


# -----------------------
# Load MNIST dataset
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# -----------------------
# Initialize models
# -----------------------
generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, 0.999)
)

# Fixed noise lets us track image quality over time.
fixed_noise = torch.randn(64, LATENT_DIM, device=device)


# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    total_loss_D = 0.0
    total_loss_G = 0.0

    for real_images, _ in loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # -----------------------
        # 1. Train discriminator
        # -----------------------
        optimizer_D.zero_grad(set_to_none=True)

        real_logits = discriminator(real_images)
        loss_real = criterion(real_logits, real_labels)

        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)

        fake_logits = discriminator(fake_images.detach())
        loss_fake = criterion(fake_logits, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # -----------------------
        # 2. Train generator
        # -----------------------
        optimizer_G.zero_grad(set_to_none=True)

        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        generated_images = generator(noise)

        # Generator wants fake images to be classified as real.
        generated_logits = discriminator(generated_images)
        loss_G = criterion(generated_logits, real_labels)

        loss_G.backward()
        optimizer_G.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    avg_loss_D = total_loss_D / len(loader)
    avg_loss_G = total_loss_G / len(loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss D: {avg_loss_D:.4f} | "
        f"Loss G: {avg_loss_G:.4f}"
    )

    # Save sample images every 5 epochs and at the end.
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == EPOCHS:
        generator.eval()

        with torch.no_grad():
            samples = generator(fixed_noise)

        utils.save_image(
            samples,
            os.path.join(
                OUTPUT_DIR,
                f"epoch_{epoch + 1:03d}.png"
            ),
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

        generator.train()


# -----------------------
# Save trained generator
# -----------------------
checkpoint_path = os.path.join(
    CHECKPOINT_DIR, "vanilla_gan.pt"
)

torch.save(
    {
        "generator_state_dict": generator.state_dict(),
        "latent_dim": LATENT_DIM,
        "image_dim": 784,
        "epochs": EPOCHS
    },
    checkpoint_path
)

print(f"Training complete. Saved generator to: {checkpoint_path}")
