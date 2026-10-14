"""
Concatenation-Based Feature Fusion
==================================

Combines feature vectors from multiple modalities by concatenating
them along the feature dimension.

Example:
    Image features: [B, 512]
    Text features:  [B, 768]
    Fused features: [B, 1280]

    After projection:
    Output features: [B, 256]

Author: Research Implementation
Framework: PyTorch
"""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Concatenation-based feature fusion.

    Args:
        input_dims (list[int]):
            Feature dimensions of each modality.
            Example: [512, 768] for image and text.

        hidden_dim (int):
            Output feature dimension after projection.

        dropout (float):
            Dropout probability. Defaults to 0.0.

        use_projection (bool):
            If True, project concatenated features to hidden_dim.
            If False, return raw concatenated features.
    """

    def __init__(
        self,
        input_dims: list[int],
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_projection: bool = True,
    ):
        super().__init__()

        if not input_dims or any(dim <= 0 for dim in input_dims):
            raise ValueError("input_dims must contain positive dimensions.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dims = input_dims
        self.total_dim = sum(input_dims)
        self.use_projection = use_projection

        if use_projection:
            if hidden_dim is None or hidden_dim <= 0:
                raise ValueError(
                    "A positive hidden_dim is required when "
                    "use_projection=True."
                )

            self.projection = nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.output_dim = hidden_dim

        else:
            self.projection = nn.Identity()
            self.output_dim = self.total_dim

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *features:
                Feature tensors with shape [B, D_i].
                All tensors must have the same batch size.

        Returns:
            Fused tensor with shape [B, output_dim].
        """

        if len(features) != len(self.input_dims):
            raise ValueError(
                f"Expected {len(self.input_dims)} inputs, "
                f"received {len(features)}."
            )

        batch_size = features[0].shape[0]

        for i, (feature, expected_dim) in enumerate(
            zip(features, self.input_dims)
        ):
            if feature.ndim != 2:
                raise ValueError(
                    f"Input {i} must have shape [B, D], "
                    f"got {tuple(feature.shape)}."
                )

            if feature.shape[0] != batch_size:
                raise ValueError("All inputs must have the same batch size.")

            if feature.shape[1] != expected_dim:
                raise ValueError(
                    f"Input {i} expected dimension {expected_dim}, "
                    f"got {feature.shape[1]}."
                )

        # Step 1: Concatenate features along the feature dimension.
        fused = torch.cat(features, dim=-1)

        # Step 2: Optionally project to the desired output dimension.
        fused = self.projection(fused)

        return fused


if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate a batch of image and text embeddings.
    batch_size = 8
    image_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 768)

    # Example 1: Concatenation + projection.
    fusion = ConcatFusion(
        input_dims=[512, 768],
        hidden_dim=256,
        dropout=0.1,
    )

    output = fusion(image_features, text_features)

    print("Image features:", image_features.shape)
    print("Text features: ", text_features.shape)
    print("Fused output:  ", output.shape)

    # Example 2: Raw concatenation without projection.
    raw_fusion = ConcatFusion(
        input_dims=[512, 768],
        use_projection=False,
    )

    raw_output = raw_fusion(image_features, text_features)
    print("Raw concatenation:", raw_output.shape)
