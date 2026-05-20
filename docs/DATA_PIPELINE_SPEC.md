# NAIME Data Pipeline Specification

This document defines the requirements for building large pretraining corpora
for NAIME models. It is written for 300GB+ raw data processing, but the same
rules should hold for future multi-terabyte corpora and larger model classes.

The goal is not only to make a dataset that can train the current V6 model. The
dataset must also be reproducible, auditable, restartable, and general enough to
support larger context windows, larger token budgets, stronger models, and later
data-mixture experiments.

## 1. Design Goals

- Training must consume a prebuilt local dataset. Training must not download,
  clean, or tokenize data on the fly.
- Every processed dataset must be traceable back to its source, license class,
  cleaning version, tokenizer version, and build command.
- Output must support deterministic resume, shuffled training, fixed validation
  isolation, and large sequential token budgets without accidental replay.
- The pipeline must be restartable at every expensive phase: download, raw
  archive, extraction, normalization, filtering, deduplication, tokenization,
  packing, validation, and final export.
- The corpus must be generic language-model data, not overfit to one model size,
  one architecture version, one local path, or one machine.

## 2. Directory Layout

Use a dedicated data root outside the repository. Do not store raw or processed
corpora in git.

```text
<DATA_ROOT>/
  raw/
    <source_name>/
      manifests/
      shards/
      logs/
  staging/
    <corpus_name>/
      normalized/
      filtered/
      dedup/
      tokenized_tmp/
      logs/
      checkpoints/
  datasets/
    <corpus_name>_ctx1024/
    <corpus_name>_ctx2048/
  manifests/
    <corpus_name>.dataset-card.json
    <corpus_name>.source-manifest.jsonl
    <corpus_name>.build-report.json
  cache/
```

Path rules:

- All scripts must accept explicit `--input`, `--output`, `--cache-dir`, and
  `--manifest` arguments.
- Paths must come from `configs/workspace.local.json`, environment variables, or
  command-line arguments. Do not hardcode local drive letters in source code.
- A dataset directory name must include source family, quality tier, token
  budget, tokenizer, and context length. Example: `rae_mix_q2_1b_gpt2_ctx1024`.

## 3. Source Intake Requirements

For each source, create one JSONL source manifest before processing any text.
Each raw shard or file gets one record.

```json
{
  "source_id": "rae/example",
  "source_name": "Example RAE Dump",
  "source_url": "https://...",
  "download_time_utc": "2026-05-18T00:00:00Z",
  "raw_path": "<DATA_ROOT>/raw/rae/shards/000001.jsonl.zst",
  "raw_bytes": 123456789,
  "sha256": "...",
  "license": "unknown|public-domain|permissive|research-only|restricted",
  "language_hint": "en|zh|multi|unknown",
  "split_policy": "train-only|can-split|validation-excluded",
  "notes": ""
}
```

Minimum intake rules:

- Compute and store SHA256 for every raw shard.
- Preserve raw files unless storage pressure is extreme. If raw files must be
  deleted, keep shard-level hashes, source URLs, sizes, and exact build logs.
- Track license and usage class. Unknown or restricted sources may be useful for
  private experiments, but they must be separable from releasable datasets.
- Keep source manifests append-only. If a source is corrected, write a new
  manifest version instead of mutating history silently.

## 4. Data Tiers

Each document should carry a quality tier. Training mixtures can then choose
different tiers without rebuilding the whole corpus.

| Tier | Meaning | Use |
|---|---|---|
| `q0_reject` | unusable, spam, binary, broken extraction, unsafe license | never train |
| `q1_low` | readable but noisy, boilerplate-heavy, weak language quality | small mixture only |
| `q2_base` | normal web/book/forum/article quality | default pretraining |
| `q3_high` | high educational, technical, literary, or explanatory quality | overweight for core runs |
| `q4_eval_holdout` | clean but held out for evaluation | never train |

The tiering system must be data-driven but inspectable. Store the individual
filter scores used to derive the tier.

## 5. Normalization

Normalize text before filtering and deduplication.

Required normalization:

- Decode as UTF-8 with explicit error handling.
- Normalize Unicode to NFC unless a source requires exact preservation.
- Standardize line endings to `\n`.
- Remove NUL bytes, control characters except tab/newline, broken replacement
  character floods, and repeated invisible separators.
- Collapse excessive blank lines while preserving paragraph boundaries.
- Strip obvious extraction wrappers, navigation boilerplate, script/style blocks,
  and machine-generated metadata that is not natural language.
- Preserve meaningful code blocks. Do not destroy indentation in code-heavy
  sources unless the source is known not to be code.

Do not perform model-specific rewriting such as adding chat templates, synthetic
speaker tags, or task prompts in the base pretraining corpus. Those belong in a
separate instruction-tuning dataset.

## 6. Document Filtering

Filtering must happen before tokenization whenever possible.

Recommended hard filters:

- Minimum text length: 256 characters for web text, lower only for curated short
  QA/dialogue sources.
- Maximum single-document length should be capped or chunked to prevent one
  source dominating token windows.
- Reject documents with extreme non-text ratio, repeated character ratio,
  punctuation-only ratio, or binary-like content.
- Reject documents with language mismatch if the target corpus is monolingual.
  For multilingual corpora, store language scores and mixture tags.
- Reject near-empty, boilerplate-only, navigation-only, OCR-garbage, and
  template-generated pages.

Recommended soft scores:

- `language_score`
- `quality_score`
- `educational_score`
- `perplexity_or_lm_score` if available
- `dedup_cluster_size`
- `source_weight`
- `safety_or_toxicity_score` if used
- `code_ratio`
- `math_ratio`
- `dialogue_ratio`

Store filter outputs in a document manifest, not only in logs.

## 7. Deduplication

Deduplication is mandatory for 300GB+ raw data. Without it, validation numbers
and generation quality become hard to trust.

Use three levels:

1. Exact document deduplication by normalized text hash.
2. Near-document deduplication by MinHash or SimHash.
3. Validation contamination check against held-out validation documents.

Dedup rules:

- Deduplicate before train/validation split when possible.
- If sources already contain official validation/test splits, keep those splits
  isolated before global deduplication.
- Never allow a document near-duplicate in train if its duplicate is in
  validation.
- Store dedup metadata: `doc_hash`, `near_hash`, `cluster_id`, `cluster_size`,
  `kept_doc_id`, and `drop_reason`.
- Use deterministic tie-breaking: prefer higher quality tier, cleaner license,
  longer text up to a cap, and stable source priority.

For large data, run dedup in shard batches and merge cluster indexes
incrementally. The process must be resumable.

## 8. Train/Validation/Test Split

Validation must be stable across architecture versions. Otherwise model changes
cannot be compared.

Rules:

- Split by document or source-level stable hash, not by packed token row.
- Keep validation completely isolated before token packing.
- Use a fixed split seed and record it in the dataset card.
- Validation should be representative, but cleaner than the noisiest training
  tail. It should not be made artificially easy.
- For 1B-token training, reserve at least 10M validation tokens if storage allows.
- For larger future corpora, reserve a fixed validation set plus optional
  domain-specific eval slices.

Recommended eval slices:

- `val_general`
- `val_educational`
- `val_code`
- `val_math`
- `val_dialogue`
- `val_long_context`
- `val_rare_domains`

## 9. Tokenization Standard

The active training code expects causal LM token rows with labels shifted by the
collator. Current large runs use GPT-2-compatible tokenization.

Requirements:

- Store tokenizer identity in the dataset card: path, model name, vocab size,
  special tokens, SHA256 of tokenizer files if local.
- Tokenize once, train many times.
- Use `add_special_tokens=False` for base pretraining unless a corpus-specific
  reason is documented.
- Insert EOS between documents or packed chunks so unrelated documents do not
  merge silently.
- Preserve source boundary metadata at least at shard/report level.
- Use `block_size = seq_len + 1` for packed rows because training uses causal
  shift. For `seq_len=1024`, store 1025-token rows unless the dataset class is
  explicitly changed.

Tokenizer compatibility:

- Current V6/100M path: GPT-2 tokenizer, `vocab_size=50257`.
- Larger future models may switch tokenizer. If tokenizer changes, rebuild the
  corpus and give it a new dataset name. Do not mix tokenizers in one training
  dataset.

## 10. Packing Rules

Packing converts tokenized documents into fixed-size causal rows.

Rules:

- Pack train and validation separately.
- Do not pack validation with training leftovers.
- Add EOS between documents.
- Record how many tokens were dropped due to final incomplete block.
- Randomization belongs in the training sampler, not in a non-reproducible packer
  unless the packer seed is stored.
- Store rows as `input_ids` with a deterministic feature type compatible with the
  installed `datasets` package. Prefer `Sequence(Value("int32"))` unless a
  specific environment requires another supported type.

Output format:

- Primary format is Hugging Face `DatasetDict.save_to_disk()`.
- Required splits: `train`, `validation`.
- Required column: `input_ids`.
- Optional columns: `source_mix_id`, `quality_tier`, `doc_count`, `token_count`
  if they do not hurt DataLoader throughput.

## 11. Data Card

Every processed dataset must include a machine-readable data card next to the
output dataset or in `<DATA_ROOT>/manifests`.

```json
{
  "name": "rae_mix_q2_1b_gpt2_ctx1024",
  "created_at_utc": "2026-05-18T00:00:00Z",
  "pipeline_version": "data-spec-v1",
  "git_commit": "...",
  "tokenizer": {
    "name": "gpt2",
    "path": "data/naime/gpt2",
    "vocab_size": 50257,
    "add_special_tokens": false,
    "eos_token_id": 50256
  },
  "block_size": 1025,
  "train_tokens": 1000000000,
  "validation_tokens": 10000000,
  "splits": {
    "train": {"rows": 975610, "tokens": 1000000000},
    "validation": {"rows": 9756, "tokens": 10000000}
  },
  "source_mix": [
    {"source_id": "rae/example", "tokens": 100000000, "ratio": 0.10, "license": "unknown"}
  ],
  "filters": {
    "min_text_chars": 256,
    "dedup": "exact+minhash",
    "language_policy": "multi"
  },
  "split_seed": 4321,
  "build_command": "...",
  "build_host": "...",
  "notes": ""
}
```

## 12. Build Stages

Split the pipeline into explicit stages. Each stage must write logs, progress,
and completion markers.

1. `download`: fetch raw shards and verify checksums.
2. `raw_index`: write source manifest and raw byte counts.
3. `normalize`: produce normalized JSONL shards.
4. `filter`: score and drop unusable documents.
5. `dedup`: exact and near-dedup.
6. `split`: stable train/validation/test assignment.
7. `tokenize`: tokenize documents into token streams.
8. `pack`: convert token streams to fixed-size rows.
9. `validate`: inspect rows, token counts, split isolation, dtype, and load speed.
10. `export`: save final HF disk dataset and data card.

Completion marker example:

```json
{
  "stage": "dedup",
  "status": "complete",
  "input_manifest_sha256": "...",
  "output_manifest_sha256": "...",
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "records_in": 123,
  "records_out": 100
}
```

If a stage fails, rerun only that stage. Do not restart from raw download unless
the raw manifest is invalid.

## 13. 300GB+ Storage and I/O Policy

Large raw data can silently fail because of disk pressure, temporary files, and
fragmented caches. Treat storage as part of the training system.

Rules:

- Keep raw, staging, and final dataset roots on the largest fast disk available.
- Maintain at least 15-20% free space during processing. For 300GB raw data, the
  working disk should ideally have 800GB-1.2TB free if raw, staging, dedup index,
  and tokenized output are kept at once.
- Use compressed raw shards where possible: `.jsonl.zst`, `.parquet`, or source
  native archives.
- Use bounded temporary directories. Do not let tokenizer or HF cache spill into
  the system drive.
- Write shards atomically: output to `*.tmp`, then rename when complete.
- Log per-stage byte throughput and document throughput.
- Avoid tiny files. Prefer shard sizes around 256MB-2GB compressed depending on
  source and filesystem behavior.
- On Windows, avoid very deep paths and excessive small-file counts where
  possible.

## 14. Quality Gates Before Training

No large training run should start until the dataset passes these checks.

Required checks:

- Dataset loads with `datasets.load_from_disk()`.
- `train` and `validation` splits exist.
- `input_ids` shape and dtype are valid.
- `len(row["input_ids"]) == seq_len + 1` for packed causal rows.
- No token IDs are negative or greater than/equal to vocab size.
- Token counts match the data card within 0.1%.
- Validation documents are not duplicated or near-duplicated in train.
- Random 100 decoded samples are readable.
- Random 100 packed rows do not show pathological repetition.
- Source mix ratios match the planned mixture.
- DataLoader can sustain acceptable throughput on the target machine.

Recommended sanity report:

```text
dataset=<path>
tokenizer=<name>
train_rows=<n>
train_tokens=<n>
validation_rows=<n>
validation_tokens=<n>
vocab_size=50257
block_size=1025
bad_token_rows=0
duplicate_train_val=0
decode_sample_pass=true
loader_rows_per_sec=<n>
```

## 15. Mixture Strategy

The data pipeline must preserve mixture control. Do not flatten all sources into
an opaque blob without statistics.

Minimum mixture metadata:

- Tokens per source.
- Tokens per license class.
- Tokens per language.
- Tokens per quality tier.
- Tokens per domain category.
- Drop counts and reasons per source.

Recommended starting policy:

- Do not allow one source to dominate more than 40-60% of the first large corpus
  unless it is intentionally a single-source experiment.
- High-quality educational/technical data can be overweighted, but keep enough
  general web/literary/dialogue data for robust generation.
- Keep code/math/dialogue slices separable so later models can adjust ratio
  without rebuilding everything.

## 16. Compatibility With NAIME

The dataset must serve the architecture rather than accidentally distort it.

NAIME-specific requirements:

- Use causal packing only. No packed row may contain future labels or auxiliary
  fields that leak future tokens into model inputs.
- Context length should be selected deliberately. Current large V6 runs use
  `seq_len=1024`, so packed rows should store 1025 tokens.
- Validation must be stable across V5/V6/VNext comparisons.
- State/world/self metrics are meaningful only when validation is not
  contaminated and token accounting is correct.
- Do not reuse older contaminated corpora as clean baselines unless they are
  explicitly marked forensic or legacy.
- Keep dataset identity in run configs and logs so checkpoint provenance is
  unambiguous.

## 17. Future-Scale Compatibility

To support higher-spec models, the corpus should be rebuildable at multiple
context lengths and token budgets.

Recommended outputs:

- A small smoke dataset: 10M-50M tokens.
- A local probe dataset: 50M-100M tokens.
- A medium dataset: 200M-500M tokens.
- A full 1B+ token dataset.
- Optional longer-context variants: ctx2048, ctx4096.

If using the same raw sources, each output must have a separate data card and
must state whether it is a prefix/subsample of another dataset or independently
sampled.

## 18. Security and Privacy

Large raw corpora can contain secrets, personal data, copyrighted fragments, and
license-incompatible material.

Minimum requirements:

- Remove obvious credentials, API keys, private tokens, local machine paths, and
  logs containing passwords.
- Do not include internal operation docs, SSH credentials, personal paths, or
  project secrets in training data.
- Keep license-restricted sources separable.
- Record whether privacy or PII filtering was applied.
- If a dataset is intended for release, apply a stricter release-grade filter and
  license audit. Private experimentation is not the same as distributable data.

## 19. Acceptance Criteria

A dataset is accepted for serious NAIME training only if all are true:

- It has a source manifest, document/filter manifest, build report, and data card.
- It can be rebuilt from raw inputs and manifests.
- It has deterministic train/validation isolation.
- It passes token, dtype, shape, decode, duplicate, and DataLoader checks.
- It has enough free-space margin for training checkpoints and logs.
- It is named and versioned clearly enough that future runs cannot confuse it
  with another corpus.
- It is compatible with current training templates without custom one-off code.

## 20. Red Flags

Stop and inspect before training if any of these occur:

- Validation loss is implausibly low from the first evaluation.
- Token count is much higher than unique source content suggests.
- Validation samples appear in training samples.
- `datasets.load_from_disk()` depends on a very specific library version because
  feature types are nonstandard.
- Dataset build logs do not identify source versions.
- Raw data and processed data have no checksums.
- A processing stage silently restarts from scratch after failure.
- Final dataset contains only one domain despite being described as mixed.
- Large numbers of rows decode as empty, repeated, or binary-like text.
