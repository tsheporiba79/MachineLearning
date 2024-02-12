from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model_args import ModelArgs
from transformer import Transformer


class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        """
        LLaMA constructor.

        Args:
            model (Transformer): The Transformer model.
            tokenizer (SentencePieceProcessor): The SentencePiece tokenizer.
            model_args (ModelArgs): Model arguments.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ) -> "LLaMA":
        """
        Build LLaMA instance.

        Args:
            checkpoints_dir (str): Directory containing model checkpoints.
            tokenizer_path (str): Path to the tokenizer model.
            load_model (bool): Whether to load pre-trained model.
            max_seq_len (int): Maximum sequence length.
            max_batch_size (int): Maximum batch size.
            device (str): Device ('cpu' or 'cuda').

        Returns:
            LLaMA: Instance of LLaMA.
        """
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"No checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text completions for given prompts.

        Args:
            prompts (List[str]): List of prompts.
            temperature (float): Softmax temperature.
            top_p (float): Top-p sampling ratio.
            max_gen_len (int): Maximum generated sequence length.

        Returns:
            List[str]: List of generated completions.
        """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        batch_size = len(prompts)
        assert (
            batch_size <= self.args.max_batch_size
        ), f"Batch size must be less than or equal to {self.args.max_batch_size}"

        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"Prompt length must be less than or equal to {self.args.max_seq_len}"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=device
        )
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            if all(eos_reached):
                break

        out_texts = [
            self.tokenizer.decode(tokens[i].tolist()) for i in range(len(tokens))
        ]
        return out_texts

    @staticmethod
    def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Sample from top-p distribution.

        Args:
            probs (torch.Tensor): Probability distribution.
            p (float): Top-p ratio.

        Returns:
            torch.Tensor: Sampled token indices.
        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "What would happen if the Earth suddenly stopped rotating?",
        "If time travel were possible, would it be ethical to visit the past?",
        # Few shot prompt
        """ Translate English to German:

            sea otter => Seeotter
            peppermint => Pfefferminze
            plush giraffe => Plüschgiraffe
            cheese =>""",
        # Zero shot prompt
        """ Tell me if the following animal is actually a Pokemon disguised as a real-life creature:
            Name: Blaziken
            Type: Fire/Fighting
            Description: Blaziken is a powerful fire-type Pokémon that resembles a large, bipedal bird with a muscular build. It is known for its blazing speed and powerful kicks, which can create explosive flames.
            Decision:
 
        """,
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    out_texts = model.text_completion(prompts, max_gen_len=64)
    for text in out_texts:
        print(text)
        print("-" * 50)
