# AT Protocol Data Pipeline

This directory contains the NAIME-compatible AT Protocol / Jetstream corpus
pipeline. It follows `docs/DATA_PIPELINE_SPEC.md`: raw capture is immutable,
processed documents remove direct social identifiers, every stage writes
manifests, and the final output is a Hugging Face `save_to_disk` dataset that
can be used by the existing trainer.

## Stages

1. `fetch-jetstream` captures public Jetstream JSON events into compressed raw
   shards.
2. `import-hf-dataset` or `import-jsonl` imports historical/bulk text sources
   directly into normalized NAIME documents.
3. `normalize` extracts post text, normalizes Unicode, redacts direct contact
   information, and replaces source identity with stable hashes.
4. `filter` applies privacy, language, length, duplication, and quality gates.
5. `dedup` removes exact duplicates and near-duplicate text buckets.
6. `split` creates deterministic train/validation/test splits by document hash.
7. `tokenize-pack` tokenizes and packs fixed-length causal-LM blocks.
8. `validate` checks the resulting disk dataset before training.

## Recommended Layout

Use a large data disk rather than the repository directory:

```text
E:\NAIME_DATA\
  raw\atproto_jetstream\
  staging\atproto_social_v1\
  datasets\atproto_social_v1_gpt2_ctx1024\
  manifests\
  logs\
```

For remote Linux/WSL-style hosts, use the same structure under a mounted data
root, for example `/data/naime`.

## Example Workflow

Set a reusable data root:

```powershell
$DataRoot = "E:\NAIME_DATA"
```

Capture a bounded Jetstream sample:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 fetch-jetstream `
  --output "$DataRoot\raw\atproto_jetstream" `
  --max-output-gib 10 `
  --shard-events 100000
```

For 10 GiB class builds, prefer a bulk dataset over live Jetstream. Live
Jetstream is useful for incremental freshness, but its post-only realtime rate
is often far too low for fast corpus construction.

Import a Hugging Face dataset in streaming mode:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 import-hf-dataset `
  --dataset "grm/three-million-bluesky-posts" `
  --split train `
  --output "$DataRoot\staging\atproto_social_v1\normalized" `
  --text-fields "text,content,body,record.text,commit.record.text" `
  --source-family atproto `
  --license research-only `
  --max-text-gib 10 `
  --progress-rows 100000 `
  --progress-seconds 60
```

Import local JSONL or JSONL.GZ shards:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 import-jsonl `
  --input "$DataRoot\raw\external_bluesky_jsonl" `
  --output "$DataRoot\staging\atproto_social_v1\normalized" `
  --text-fields "text,content,body,record.text,commit.record.text" `
  --source-name "external-bluesky-jsonl" `
  --license research-only `
  --max-text-gib 10 `
  --progress-rows 100000 `
  --progress-seconds 60
```

Normalize raw shards:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 normalize `
  --input "$DataRoot\raw\atproto_jetstream" `
  --output "$DataRoot\staging\atproto_social_v1\normalized"
```

Filter and tier documents:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 filter `
  --input "$DataRoot\staging\atproto_social_v1\normalized" `
  --output "$DataRoot\staging\atproto_social_v1\filtered" `
  --min-text-chars 80 `
  --max-text-chars 12000
```

Deduplicate:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 dedup `
  --input "$DataRoot\staging\atproto_social_v1\filtered" `
  --output "$DataRoot\staging\atproto_social_v1\dedup"
```

Create deterministic splits:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 split `
  --input "$DataRoot\staging\atproto_social_v1\dedup" `
  --output "$DataRoot\staging\atproto_social_v1\splits" `
  --validation-ratio 0.01 `
  --test-ratio 0.002 `
  --seed naime-atproto-v1
```

Tokenize and pack:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 tokenize-pack `
  --input "$DataRoot\staging\atproto_social_v1\splits" `
  --output "$DataRoot\datasets\atproto_social_v1_gpt2_ctx1024" `
  --tokenizer-path "G:\Program\naime-hybrid-moe\data\naime\gpt2" `
  --block-size 1025 `
  --train-tokens 50000000 `
  --validation-tokens 2000000
```

Validate:

```powershell
.\scripts\data_atproto\run_atproto_pipeline.ps1 validate `
  --dataset "$DataRoot\datasets\atproto_social_v1_gpt2_ctx1024" `
  --tokenizer-path "G:\Program\naime-hybrid-moe\data\naime\gpt2"
```

## Scaling Notes

For 100 GiB class raw ingestion, run `fetch-jetstream` continuously with shard
rotation and preserve the raw manifest. Normalize/filter/dedup can be rerun
stage by stage after interruption because each stage writes completion markers
and does not mutate prior outputs.

AT Protocol social text should usually be a mixture component, not the only
pretraining corpus. A safe first target is 5-15% of a broader FineWeb-Edu style
mix, unless evaluation shows that social-dialogue adaptation is the main goal.

## Privacy And License Policy

The default license label is `research-only` because public social data may
carry platform terms and user expectations that are stricter than ordinary web
pages. Processed documents deliberately avoid DIDs, handles, record keys, raw
timestamps, emails, phone numbers, and secrets. Do not publish raw shards or a
processed dataset without a separate legal and privacy review.
