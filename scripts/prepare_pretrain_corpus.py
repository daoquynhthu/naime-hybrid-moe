import argparse
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a token-budgeted NAIME pretraining corpus from HF datasets.")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer-path", default="data/naime/gpt2")
    parser.add_argument("--output", default="data/naime/fineweb_edu_50m")
    parser.add_argument(
        "--block-size", type=int, default=513, help="Stored tokens per row. Training uses block_size - 1."
    )
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--validation-tokens", type=int, default=2_000_000)
    parser.add_argument("--min-score", type=float, default=None, help="Optional FineWeb-Edu score filter.")
    parser.add_argument("--min-language-score", type=float, default=None)
    parser.add_argument("--min-text-chars", type=int, default=128)
    parser.add_argument("--max-text-chars", type=int, default=200_000)
    parser.add_argument("--validation-first-docs", type=int, default=10_000)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _features() -> Features:
    return Features(
        {
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int32")),
        }
    )


def _passes_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    text = row.get(args.text_column)
    if not isinstance(text, str):
        return False
    text_len = len(text)
    if text_len < args.min_text_chars or text_len > args.max_text_chars:
        return False
    if args.min_score is not None and row.get("score") is not None and float(row["score"]) < args.min_score:
        return False
    if (
        args.min_language_score is not None
        and row.get("language_score") is not None
        and float(row["language_score"]) < args.min_language_score
    ):
        return False
    return True


def _load_stream(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    dataset_kwargs: dict[str, Any] = {
        "path": args.dataset_name,
        "split": args.dataset_split,
        "streaming": args.streaming,
    }
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    return load_dataset(**dataset_kwargs)


def _chunk_generator(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    target_tokens: int,
) -> Iterator[dict[str, list[int]]]:
    eos = tokenizer.eos_token_id
    buffer: list[int] = []
    emitted_tokens = 0
    for row in rows:
        if not _passes_filters(row, args):
            continue
        ids = tokenizer(row[args.text_column], add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buffer.extend(ids)
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= args.block_size and emitted_tokens < target_tokens:
            chunk = buffer[: args.block_size]
            del buffer[: args.block_size]
            emitted_tokens += args.block_size
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * args.block_size,
                "labels": chunk.copy(),
            }
        if emitted_tokens >= target_tokens:
            return


def _count_tokens(dataset: Dataset, block_size: int) -> int:
    return len(dataset) * block_size


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to rebuild it")
        shutil.rmtree(output)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.dataset_name} config={args.dataset_config or '<default>'} split={args.dataset_split}")
    print(f"Building validation from first {args.validation_first_docs:,} streamed docs ...")
    validation_rows = _load_stream(args).take(args.validation_first_docs)
    validation = Dataset.from_generator(
        lambda: _chunk_generator(validation_rows, tokenizer, args, args.validation_tokens),
        features=_features(),
    )

    print("Building train from stream after validation prefix ...")
    train_rows = _load_stream(args).skip(args.validation_first_docs)
    train = Dataset.from_generator(
        lambda: _chunk_generator(train_rows, tokenizer, args, args.train_tokens),
        features=_features(),
    )

    dataset = DatasetDict({"train": train, "validation": validation})
    dataset.save_to_disk(str(output))
    print(f"Saved processed dataset to {output}")
    for split, split_ds in dataset.items():
        print(f"{split}: rows={len(split_ds):,} tokens={_count_tokens(split_ds, args.block_size):,}")


if __name__ == "__main__":
    main()
