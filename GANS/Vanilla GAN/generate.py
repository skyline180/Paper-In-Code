
import argparse
import os

import torch
from torchvision.utils import save_image

from model import Generator


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using a trained Vanilla GAN."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/vanilla_gan.pt",
        help="Path to the trained generator checkpoint."
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=64,
        help="Number of images to generate."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/generated.png",
        help="Path to save the generated image grid."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible noise."
    )

    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be at least 1.")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train the model first using train.py."
        )

    # Load checkpoint saved by train.py.
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    latent_dim = checkpoint["latent_dim"]

    generator = Generator(
        latent_dim=latent_dim,
        image_dim=checkpoint["image_dim"]
    ).to(device)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )
    generator.eval()

    # Sample new random latent vectors.
    noise = torch.randn(
        args.num_images,
        latent_dim,
        device=device
    )

    with torch.no_grad():
        generated_images = generator(noise)

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    columns = min(8, args.num_images)

    save_image(
        generated_images,
        args.output,
        nrow=columns,
        normalize=True,
        value_range=(-1, 1)
    )

    print(f"Generated {args.num_images} images.")
    print(f"Saved image grid to: {args.output}")


if __name__ == "__main__":
    main()
