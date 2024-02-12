import torch
import torch.nn as nn
from .Layers import LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock


class ResidualConnection(nn.Module):
    """A module implementing the residual connection mechanism.

    Args:
        features (int): The number of input features.
        dropout (float): Dropout probability.

    Attributes:
        dropout (torch.nn.Dropout): Dropout layer.
        norm (Layers.LayerNormalization): Layer normalization module.
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): Sublayer function.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """An encoder block in the Transformer architecture.

    Args:
        features (int): The number of input features.
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        feed_forward_block (FeedForwardBlock): Feed-forward block.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        feed_forward_block (FeedForwardBlock): Feed-forward block.
        residual_connections (torch.nn.ModuleList): List of residual connections.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            source_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, source_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class DecoderBlock(nn.Module):
    """A decoder block in the Transformer architecture.

    Args:
        features (int): The number of input features.
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        feed_forward_block (FeedForwardBlock): Feed-forward block.
        dropout (float): Dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        feed_forward_block (FeedForwardBlock): Feed-forward block.
        residual_connections (torch.nn.ModuleList): List of residual connections.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            source_mask (torch.Tensor): Source mask tensor.
            target_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, source_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
