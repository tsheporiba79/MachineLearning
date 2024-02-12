import torch
import logging
import sys
from pathlib import Path
from tokenizers import Tokenizer
from ModelConfig import get_config, latest_weights_file_path
from Transformer import create_transformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"


def translate_sentence(
    sentence: str,
    model: torch.nn.Module,
    tokenizer_source: Tokenizer,
    tokenizer_target: Tokenizer,
    device: torch.device,
) -> str:
    """Translate a single sentence using the provided model and tokenizers.

    Args:
        sentence (str): The sentence to translate.
        model (torch.nn.Module): The translation model.
        tokenizer_source (Tokenizer): Tokenizer for the source language.
        tokenizer_target (Tokenizer): Tokenizer for the target language.
        device (torch.device): The device to run the model on.

    Returns:
        str: The translated sentence.
    """
    # Preprocess the input sentence
    source = tokenizer_source.encode(sentence)
    source_ids = torch.tensor(source.ids, dtype=torch.long, device=device)
    seq_len = len(source.ids)

    # Translate the sentence
    model.eval()
    with torch.no_grad():
        # Encode the source sentence
        source_input = torch.cat(
            [
                torch.tensor(
                    [tokenizer_source.token_to_id(SOS_TOKEN)],
                    dtype=torch.long,
                    device=device,
                ),
                source_ids,
                torch.tensor(
                    [tokenizer_source.token_to_id(EOS_TOKEN)],
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(
                    [tokenizer_source.token_to_id(PAD_TOKEN)]
                    * (seq_len - len(source.ids) - 2),
                    dtype=torch.long,
                    device=device,
                ),
            ]
        ).unsqueeze(0)
        source_mask = (
            (source_input != tokenizer_source.token_to_id(PAD_TOKEN))
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            .to(device)
        )
        encoder_output = model.encode(source_input, source_mask)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.tensor(
            [[tokenizer_target.token_to_id(SOS_TOKEN)]], device=device
        )

        translated_sentence = []

        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # Build mask for target and calculate output
            decoder_mask = (
                torch.triu(
                    torch.ones((1, decoder_input.size(1), decoder_input.size(1))),
                    diagonal=1,
                )
                .type(torch.int)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # Project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)

            # Append the translated word
            translated_sentence.append(tokenizer_target.decode([next_word.item()]))

            # Break if we predict the end of sentence token
            if next_word == tokenizer_target.token_to_id(EOS_TOKEN):
                break

    return " ".join(translated_sentence)


def translate(sentence: str) -> None:
    """Translate a given sentence.

    Args:
        sentence (str): The sentence to translate.
    """
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    config = get_config()
    tokenizer_source = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_source"])))
    )
    tokenizer_target = Tokenizer.from_file(
        str(Path(config["tokenizer_file"].format(config["lang_target"])))
    )

    model = create_transformer(
        tokenizer_source.get_vocab_size(),
        tokenizer_target.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])

    # Translate the sentence
    translation = translate_sentence(
        sentence, model, tokenizer_source, tokenizer_target, device
    )
    logger.info("Input sentence: %s", sentence)
    logger.info("Translated sentence: %s", translation)


if __name__ == "__main__":
    sentence = (
        sys.argv[1] if len(sys.argv) > 1 else "I am trying to translate this sentence."
    )
    translate(sentence)
