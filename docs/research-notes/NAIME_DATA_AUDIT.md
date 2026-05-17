# NAIME Data Audit

Date: 2026-05-09

Copied from:

```text
<NAIME_DATA_SOURCE>
```

Copied to:

```text
<PROJECT_ROOT>\data\naime
```

## Contents

Total copied size:

```text
603,355,589 bytes
32 files
```

Top-level directories:

- `gpt2`: local GPT-2 config, tokenizer files, and `model.safetensors`.
- `wikitext`: raw HuggingFace WikiText dataset cache/save.
- `wikitext_processed`: tokenized and grouped WikiText dataset.

## GPT-2 Local Model

Path:

```text
data\naime\gpt2
```

Important facts:

- model type: GPT-2;
- vocab size: 50257;
- context: 1024;
- hidden size: 768;
- layers: 12;
- heads: 12.

This is useful later if we want to reproduce the original NAIME setup with a frozen GPT-2 backbone.

## Processed WikiText

Path:

```text
data\naime\wikitext_processed
```

Loaded as:

```python
from datasets import load_from_disk
ds = load_from_disk("data/naime/wikitext_processed")
```

Splits:

```text
train: 2318 rows
validation: 240 rows
test: 274 rows
```

Fields:

- `input_ids`
- `attention_mask`
- `labels`

Observed row length:

```text
1024 tokens per row
```

## Training Use

The saved `labels` mirror `input_ids`, matching standard HuggingFace causal-LM convention where the model/loss performs the shift internally.

Our training loss expects already-shifted labels, so the new `HFDiskCausalDataset` wrapper performs:

```text
input_ids = tokens[:seq_len]
labels    = tokens[1:seq_len+1]
```

Recommended smoke command:

```powershell
.\scripts\train.ps1 --architecture naime_state_moe --run-name naime_wikitext_smoke --data-path data\naime\wikitext_processed --data-format hf_disk --data-split train --vocab-size 50257 --max-steps 10 --batch-size 2 --seq-len 128
```

## Interpretation

This dataset is appropriate for first-stage language-modeling experiments because it gives us:

- GPT-2-tokenized text;
- fixed-size 1024-token blocks;
- train/validation/test splits;
- no immediate need to build a tokenizer pipeline.

It is small enough for quick iteration, but not large enough to prove final architecture quality. Its best role is early stability and baseline comparison.
