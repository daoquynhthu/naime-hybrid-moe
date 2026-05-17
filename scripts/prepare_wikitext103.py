import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and tokenize WikiText-103 for NAIME training.")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer-path", default="data/naime/gpt2")
    parser.add_argument("--raw-output", default="data/naime/wikitext103_raw")
    parser.add_argument("--processed-output", default="data/naime/wikitext103_processed")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--tokenize-batch-size", type=int, default=1000)
    parser.add_argument("--group-batch-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_output = Path(args.raw_output)
    processed_output = Path(args.processed_output)

    if processed_output.exists() and not args.overwrite:
        raise FileExistsError(f"{processed_output} already exists; pass --overwrite to rebuild it")
    if raw_output.exists() and not args.overwrite:
        print(f"Using existing raw dataset at {raw_output}")
        dataset = DatasetDict.load_from_disk(str(raw_output))
    else:
        print(f"Downloading {args.dataset_name}/{args.dataset_config} ...")
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        if raw_output.exists():
            import shutil

            shutil.rmtree(raw_output)
        dataset.save_to_disk(str(raw_output))
        print(f"Saved raw dataset to {raw_output}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [text for text in batch["text"] if text and text.strip()]
        if not texts:
            return {"input_ids": []}
        return {"input_ids": tokenizer(texts, add_special_tokens=False)["input_ids"]}

    def group_batch(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        eos = tokenizer.eos_token_id
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
            if eos is not None:
                concatenated.append(eos)
        total_length = (len(concatenated) // args.block_size) * args.block_size
        chunks = [concatenated[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * args.block_size for _ in chunks],
            "labels": [chunk.copy() for chunk in chunks],
        }

    print("Tokenizing text ...")
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.tokenize_batch_size,
        remove_columns=["text"],
        desc="tokenize",
    )

    print(f"Grouping into {args.block_size}-token blocks ...")
    processed = tokenized.map(
        group_batch,
        batched=True,
        batch_size=args.group_batch_size,
        desc="group",
    )

    if processed_output.exists():
        import shutil

        shutil.rmtree(processed_output)
    processed.save_to_disk(str(processed_output))
    print(f"Saved processed dataset to {processed_output}")
    for split, split_ds in processed.items():
        tokens = len(split_ds) * args.block_size
        print(f"{split}: rows={len(split_ds):,} tokens={tokens:,}")


if __name__ == "__main__":
    main()
