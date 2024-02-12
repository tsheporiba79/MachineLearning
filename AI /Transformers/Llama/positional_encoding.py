import torch


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    """Precompute the positional encodings.

    Args:
        head_dim (int): Dimension of the head.
        seq_len (int): Length of the sequence.
        device (str): Device for computation.
        theta (float, optional): Theta parameter. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed positional encodings.
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(
    x: torch.Tensor, freqs_complex: torch.Tensor, device: str
) -> torch.Tensor:
    """Apply rotary embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        freqs_complex (torch.Tensor): Precomputed positional encodings.
        device (str): Device for computation.

    Returns:
        torch.Tensor: Tensor with applied rotary embeddings.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
