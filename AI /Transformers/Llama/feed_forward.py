import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_args import ModelArgs


class FeedForward(nn.Module):
    """Feed-forward network component of the transformer encoder block."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the FeedForward module.

        Args:
            args (ModelArgs): Model arguments.
        """
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
