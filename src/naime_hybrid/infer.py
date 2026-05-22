import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import torch

from naime_hybrid.config import NAIMEStateMoEConfig
from naime_hybrid.models import build_model
from naime_hybrid.training.checkpoint import normalize_state_dict_for_model
from naime_hybrid.training.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from any NAIME Hybrid run checkpoint.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing config.json.")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help=(
            "Checkpoint filename/path, or 'auto'. Auto prefers models/model_best.pt, then best.pt, "
            "models/model_latest.pt, latest.pt."
        ),
    )
    parser.add_argument("--prompt", default="", help="Prompt text.")
    parser.add_argument("--prompt-file", default=None, help="Read prompt text from a UTF-8 file.")
    parser.add_argument(
        "--prompt-jsonl",
        default=None,
        help="Optional JSONL file with a text/prompt field; each row is generated independently.",
    )
    parser.add_argument("--tokenizer-path", default="data/naime/gpt2", help="HF tokenizer path for GPT-style models.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--show-aux", action="store_true", help="Print a compact auxiliary-state summary.")
    parser.add_argument("--jsonl-output", default=None, help="Write generated rows as JSONL.")
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


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        seen = torch.unique(generated[batch_idx])
        selected = logits[batch_idx, seen]
        logits[batch_idx, seen] = torch.where(selected < 0, selected * penalty, selected / penalty)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    generated: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    logits = logits.float()
    logits = apply_repetition_penalty(logits, generated, repetition_penalty)
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


def resolve_checkpoint(run_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name != "auto":
        checkpoint_path = Path(checkpoint_name)
        return checkpoint_path if checkpoint_path.is_absolute() else run_dir / checkpoint_path

    candidates = [
        run_dir / "models" / "model_best.pt",
        run_dir / "best.pt",
        run_dir / "models" / "model_latest.pt",
        run_dir / "latest.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint found in {run_dir}; tried: " + ", ".join(str(p.relative_to(run_dir)) for p in candidates)
    )


def _config_from_dict(raw: dict[str, Any]) -> NAIMEStateMoEConfig:
    valid = {field.name for field in fields(NAIMEStateMoEConfig)}
    return NAIMEStateMoEConfig(**{key: value for key, value in raw.items() if key in valid})


def _load_json_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json in {run_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_run(run_dir: Path, checkpoint_name: str, device: torch.device):
    train_config = _load_json_config(run_dir)
    checkpoint_path = resolve_checkpoint(run_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_config, dict):
        train_config = {**train_config, **checkpoint_config}
    architecture = train_config.get("architecture", "naime_state_moe")
    model_config_raw = train_config.get("model", train_config)
    model_config = _config_from_dict(model_config_raw)
    model = build_model(architecture, model_config).to(device)

    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    state = normalize_state_dict_for_model(model, state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, model_config, train_config, checkpoint_path


def iter_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompt_jsonl:
        rows = []
        with Path(args.prompt_jsonl).open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("prompt", row.get("text", ""))
                rows.append({"id": row.get("id", line_no), "prompt": str(text)})
        return rows
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    return [{"id": 1, "prompt": prompt}]


def summarize_aux(aux: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}

    def add(prefix: str, obj: Any) -> None:
        if torch.is_tensor(obj):
            if obj.numel() == 0:
                return
            values.setdefault(prefix, []).append(float(obj.detach().float().mean().cpu()))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                add(f"{prefix}.{key}" if prefix else str(key), value)

    for layer in aux:
        add("", layer)
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def generate_one(
    *,
    model,
    model_config: NAIMEStateMoEConfig,
    tokenizer,
    prompt: str,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
) -> tuple[str, dict[str, float]]:
    input_ids = encode_prompt(tokenizer, prompt, device)
    generated = input_ids.clone()
    last_aux: dict[str, float] = {}

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = generated[:, -model_config.max_seq_len :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                # Generation has no padding. Keep infer_pad_mask disabled so token id 0 remains valid
                # for GPT-style tokenizers instead of being treated as a pad token.
                output = model(context, return_aux=args.show_aux, infer_pad_mask=False)
                logits = output["logits"][:, -1, :]
            if args.show_aux and "aux" in output:
                last_aux = summarize_aux(output["aux"])
            next_token = sample_next_token(
                logits,
                generated,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
            )
            generated = torch.cat([generated, next_token], dim=1)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and int(next_token.item()) == int(eos):
                break

    return decode_tokens(tokenizer, generated[0].detach().cpu().tolist()), last_aux


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"

    model, model_config, train_config, checkpoint_path = load_run(run_dir, args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer_path, model_config.vocab_size)

    print(f"run: {run_dir}")
    print(f"architecture: {train_config.get('architecture', 'unknown')}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"model: layers={model_config.n_layers} d_model={model_config.d_model} vocab={model_config.vocab_size}")
    print("----")

    output_handle = Path(args.jsonl_output).open("w", encoding="utf-8") if args.jsonl_output else None
    try:
        prompts = iter_prompts(args)
        sample_idx = 0
        for row in prompts:
            for _ in range(args.num_samples):
                sample_idx += 1
                text, aux_summary = generate_one(
                    model=model,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    prompt=row["prompt"],
                    device=device,
                    use_amp=use_amp,
                    args=args,
                )
                payload = {
                    "id": row["id"],
                    "sample": sample_idx,
                    "prompt": row["prompt"],
                    "text": text,
                    "aux": aux_summary,
                }
                if output_handle is not None:
                    output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(f"[{sample_idx}] prompt_id={row['id']}")
                print(text)
                if args.show_aux and aux_summary:
                    keep = {key: round(value, 4) for key, value in sorted(aux_summary.items()) if key.startswith("v")}
                    print("aux:", json.dumps(keep, ensure_ascii=False, sort_keys=True))
                print("----")
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
