import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the projection layer.

        Args:
            d_model (int): The dimensionality of the model's output.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, vocab_size).
        """
        return self.proj(x)
