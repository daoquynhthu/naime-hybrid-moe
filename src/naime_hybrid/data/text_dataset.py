from pathlib import Path

import torch
from torch.utils.data import Dataset

from naime_hybrid.training.losses import IGNORE_INDEX


class ByteTextDataset(Dataset):
    """Simple byte-level LM dataset for early architecture experiments."""

    def __init__(self, text: str | bytes, seq_len: int, max_samples: int | None = None):
        if isinstance(text, str):
            data = text.encode("utf-8", errors="ignore")
        else:
            data = text
        if len(data) < seq_len + 1:
            raise ValueError("dataset text is shorter than seq_len + 1")

        self.tokens = torch.tensor(list(data), dtype=torch.long) + 1
        self.seq_len = seq_len
        self.max_start = len(self.tokens) - seq_len - 1
        self.max_samples = max_samples

    @classmethod
    def from_file(cls, path: str | Path, seq_len: int, max_samples: int | None = None):
        content = Path(path).read_bytes()
        return cls(content, seq_len=seq_len, max_samples=max_samples)

    @property
    def vocab_size(self) -> int:
        return 257

    def __len__(self) -> int:
        natural = self.max_start + 1
        return min(natural, self.max_samples) if self.max_samples is not None else natural

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = idx % (self.max_start + 1)
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


class RandomTokenDataset(Dataset):
    """Deterministic random data for smoke tests."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int, seed: int = 1234):
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(1, vocab_size, (num_samples, seq_len), generator=generator)
        self.labels = torch.randint(1, vocab_size, (num_samples, seq_len), generator=generator)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


class HFDiskCausalDataset(Dataset):
    """Wrapper for HuggingFace datasets saved with input_ids.

    The NAIME WikiText data stores 1024-token blocks with labels equal to
    input_ids. This wrapper performs causal shifting for our training loss:
    input = tokens[:-1], label = tokens[1:].
    """

    def __init__(
        self,
        path: str | Path,
        split: str,
        seq_len: int,
        max_samples: int | None = None,
    ):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError("datasets is required for HFDiskCausalDataset") from exc

        dataset_dict = load_from_disk(str(path))
        if split not in dataset_dict:
            raise ValueError(f"split {split!r} not found; available: {list(dataset_dict.keys())}")
        self.dataset = dataset_dict[split]
        self.seq_len = seq_len
        self.max_samples = max_samples

        if "input_ids" not in self.dataset.features:
            raise ValueError("HF disk dataset must contain input_ids")
        # Ask datasets to materialize input_ids as torch tensors up front so
        # workers avoid rebuilding tensors sample-by-sample in Python.
        self.dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=False)

    def __len__(self) -> int:
        natural = len(self.dataset)
        return min(natural, self.max_samples) if self.max_samples is not None else natural

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.dataset[int(idx)]
        ids = row["input_ids"]
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, dtype=torch.long)
        else:
            ids = ids.to(dtype=torch.long)
        if ids.numel() < 2:
            raise ValueError("sample is too short for causal shift")
        return {"input_ids": ids}

    @staticmethod
    def causal_collate(
        batch: list[dict[str, torch.Tensor]], seq_len: int, pad_token_id: int = 0
    ) -> dict[str, torch.Tensor]:
        ids_list = [item["input_ids"] for item in batch]

        lengths = [ids.numel() for ids in ids_list]
        usable = min(min(lengths) - 1, seq_len)
        need = usable + 1

        padded = []
        for ids in ids_list:
            if ids.numel() >= need:
                padded.append(ids[:need])
            else:
                padded.append(torch.nn.functional.pad(ids, (0, need - ids.numel()), value=pad_token_id))

        stacked = torch.stack(padded)
        input_ids = stacked[:, :usable]
        labels = stacked[:, 1:need]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if any(ids.numel() < need for ids in ids_list):
            input_lengths = torch.tensor([min(max(ids.numel() - 1, 0), usable) for ids in ids_list])
            positions = torch.arange(usable).unsqueeze(0)
            attention_mask = positions < input_lengths.unsqueeze(1)
            labels = labels.masked_fill(~attention_mask, IGNORE_INDEX)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
