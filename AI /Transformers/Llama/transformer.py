import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .model_args import ModelArgs
from .encoder_block import EncoderBlock
from .positional_encoding import precompute_theta_pos_frequencies


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        """Transformer model.

        Args:
            args (ModelArgs): Model arguments.
        """
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Forward pass of the transformer model.

        Args:
            tokens (torch.Tensor): Input tokens.
            start_pos (int): Start position.

        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        h = self.tok_embeddings(tokens)
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
