import torch
import torch.nn as nn
from .EncoderandDecoder import Encoder, Decoder
from .Embeddings import InputEmbeddings, PositionalEncoding
from .Blocks import EncoderBlock, DecoderBlock
from .Layers import FeedForwardBlock, MultiHeadAttentionBlock
from .Projection import ProjectionLayer


class Transformer(nn.Module):
    """
    Transformer model architecture.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        source_embed (InputEmbeddings): Source language embeddings.
        target_embed (InputEmbeddings): Target language embeddings.
        source_pos (PositionalEncoding): Positional encoding for source language.
        target_pos (PositionalEncoding): Positional encoding for target language.
        projection_layer (ProjectionLayer): Projection layer.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        source_embed (InputEmbeddings): Source language embeddings.
        target_embed (InputEmbeddings): Target language embeddings.
        source_pos (PositionalEncoding): Positional encoding for source language.
        target_pos (PositionalEncoding): Positional encoding for target language.
        projection_layer (ProjectionLayer): Projection layer.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_embed: InputEmbeddings,
        target_embed: InputEmbeddings,
        source_pos: PositionalEncoding,
        target_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode the source input.

        Args:
            source (torch.Tensor): Source input tensor.
            source_mask (torch.Tensor): Mask for source input.

        Returns:
            torch.Tensor: Encoded source tensor.
        """
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the target input.

        Args:
            encoder_output (torch.Tensor): Encoder output tensor.
            source_mask (torch.Tensor): Mask for source input.
            target (torch.Tensor): Target input tensor.
            target_mask (torch.Tensor): Mask for target input.

        Returns:
            torch.Tensor: Decoded target tensor.
        """
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected tensor.
        """
        return self.projection_layer(x)


def create_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Create a Transformer model.

    Args:
        source_vocab_size (int): Size of source vocabulary.
        target_vocab_size (int): Size of target vocabulary.
        source_seq_len (int): Length of source sequence.
        target_seq_len (int): Length of target sequence.
        d_model (int, optional): Dimension of model. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimension of feedforward layer. Defaults to 2048.

    Returns:
        Transformer: Created Transformer model.
    """
    # Create embeddings and positional encodings
    source_embed = InputEmbeddings(d_model, source_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)
    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = [
        EncoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        for _ in range(N)
    ]
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create decoder blocks
    decoder_blocks = [
        DecoderBlock(
            d_model,
            MultiHeadAttentionBlock(d_model, h, dropout),
            MultiHeadAttentionBlock(d_model, h, dropout),
            FeedForwardBlock(d_model, d_ff, dropout),
            dropout,
        )
        for _ in range(N)
    ]
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    # Create transformer
    transformer = Transformer(
        encoder,
        decoder,
        source_embed,
        target_embed,
        source_pos,
        target_pos,
        projection_layer,
    )

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
