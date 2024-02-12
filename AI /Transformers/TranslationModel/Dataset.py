import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ModelDataset(Dataset):
    """Custom dataset for model training."""

    def __init__(
        self,
        ds: Dataset,
        tokenizer_source: PreTrainedTokenizer,
        tokenizer_target: PreTrainedTokenizer,
        source_lang: str,
        target_lang: str,
        seq_len: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            ds (Dataset): Original dataset.
            tokenizer_source (PreTrainedTokenizer): Tokenizer for the source language.
            tokenizer_target (PreTrainedTokenizer): Tokenizer for the target language.
            source_lang (str): Source language code.
            target_lang (str): Target language code.
            seq_len (int): Maximum sequence length.
        """
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.sos_token_id = tokenizer_target.convert_tokens_to_ids("[SOS]")
        self.eos_token_id = tokenizer_target.convert_tokens_to_ids("[EOS]")
        self.pad_token_id = tokenizer_target.convert_tokens_to_ids("[PAD]")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: Dictionary containing encoder input, decoder input, masks, labels, source text, and target text.
        """
        source_target_pair = self.ds[idx]
        source_text = source_target_pair["translation"][self.source_lang]
        target_text = source_target_pair["translation"][self.target_lang]

        enc_input_tokens = self.tokenizer_source.encode(
            source_text,
            add_special_tokens=True,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
        ).input_ids
        dec_input_tokens = self.tokenizer_target.encode(
            target_text, add_special_tokens=False
        )

        enc_num_padding_tokens = max(0, self.seq_len - len(enc_input_tokens))
        dec_num_padding_tokens = max(0, self.seq_len - len(dec_input_tokens) - 1)

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("This sentence exceeds the maximum allowed length")

        encoder_input = torch.tensor(
            enc_input_tokens
            + [self.eos_token_id]
            + [self.pad_token_id] * enc_num_padding_tokens,
            dtype=torch.long,
        )
        decoder_input = torch.tensor(
            [self.sos_token_id]
            + dec_input_tokens
            + [self.pad_token_id] * dec_num_padding_tokens,
            dtype=torch.long,
        )
        label = torch.tensor(
            dec_input_tokens
            + [self.eos_token_id]
            + [self.pad_token_id] * dec_num_padding_tokens,
            dtype=torch.long,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (
            (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int()
        )
        decoder_mask = (decoder_input != self.pad_token_id).unsqueeze(
            0
        ).int() & causal_mask(decoder_input.size(0))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }


def causal_mask(size: int) -> torch.Tensor:
    """Generate a causal mask.

    Args:
        size (int): Size of the mask.

    Returns:
        torch.Tensor: Causal mask tensor.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.bool)
    return ~mask
