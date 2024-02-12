import torch
import torch.nn as nn
from .Layers import LayerNormalization


class Encoder(nn.Module):
    """Encoder module of the Transformer model."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """Initialize the Encoder module.

        Args:
            features (int): The number of features.
            layers (nn.ModuleList): List of encoder layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Encoder module.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Decoder module of the Transformer model."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """Initialize the Decoder module.

        Args:
            features (int): The number of features.
            layers (nn.ModuleList): List of decoder layers.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Decoder module.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            source_mask (torch.Tensor): Source mask tensor.
            target_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)
