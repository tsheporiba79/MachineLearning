import torch
import torch.nn as nn
from .self_attention import SelfAttention
from .feed_forward import FeedForward
from .rms_norm import RMSNorm
from .model_args import ModelArgs


class EncoderBlock(nn.Module):
    """Encoder block in a transformer architecture."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the EncoderBlock.

        Args:
            args (ModelArgs): Model arguments.
        """
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position.
            freqs_complex (torch.Tensor): Complex frequencies.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
