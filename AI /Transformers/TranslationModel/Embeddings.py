import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """Input Embeddings module for the Transformer model."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize the InputEmbeddings module.

        Args:
            d_model (int): The embedding dimension.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the InputEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded output tensor.
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Positional Encoding module for the Transformer model."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """Initialize the PositionalEncoding module.

        Args:
            d_model (int): The embedding dimension.
            seq_len (int): The maximum sequence length.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.shape[1], :].detach()
        return self.dropout(x)
