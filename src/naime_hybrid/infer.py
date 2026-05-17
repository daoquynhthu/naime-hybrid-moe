import argparse
import json
from pathlib import Path

import torch

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models import build_model
from naime_hybrid.training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from any NAIME Hybrid run checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing config.json.")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename or path.")
    parser.add_argument("--prompt", default="", help="Prompt text.")
    parser.add_argument("--prompt-file", default=None, help="Read prompt text from a UTF-8 file.")
    parser.add_argument("--tokenizer-path", default="data/naime/gpt2", help="HF tokenizer path for GPT-style models.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


class ByteCodec:
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [b + 1 for b in text.encode("utf-8", errors="ignore")]

    def decode(self, ids: list[int]) -> str:
        data = bytes(max(0, min(255, int(i) - 1)) for i in ids if int(i) > 0)
        return data.decode("utf-8", errors="ignore")


def load_tokenizer(tokenizer_path: str, vocab_size: int):
    if vocab_size <= 257:
        return ByteCodec()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for GPT-style text generation") from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_prompt(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    if isinstance(tokenizer, ByteCodec):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        eos = getattr(tokenizer, "eos_token_id", None)
        ids = [eos if eos is not None else 1]
    return torch.tensor([ids], dtype=torch.long, device=device)


def decode_tokens(tokenizer, ids: list[int]) -> str:
    if isinstance(tokenizer, ByteCodec):
        return tokenizer.decode(ids)
    return tokenizer.decode(ids, skip_special_tokens=True)


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def load_run(run_dir: Path, checkpoint_name: str, device: torch.device):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json in {run_dir}")
    train_config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = NAIMEStateMoEConfig(**train_config["model"])
    model = build_model(train_config["architecture"], model_config).to(device)

    checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    # Strip torch.compile _orig_mod prefix if present (compiled training saves wrapped keys)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_config, train_config, checkpoint_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, model_config, train_config, checkpoint_path = load_run(run_dir, args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer_path, model_config.vocab_size)
    input_ids = encode_prompt(tokenizer, prompt, device)
    generated = input_ids.clone()

    print(f"run: {run_dir}")
    print(f"architecture: {train_config['architecture']}")
    print(f"checkpoint: {checkpoint_path}")
    print("----")

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = generated[:, -model_config.max_seq_len :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(context, return_aux=False)["logits"][:, -1, :]
            next_token = sample_next_token(logits, args.temperature, args.top_k, args.top_p)
            generated = torch.cat([generated, next_token], dim=1)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and int(next_token.item()) == int(eos):
                break

    print(decode_tokens(tokenizer, generated[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
