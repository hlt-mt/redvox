# Qwen3Guard Inference & Evaluation

Safety evaluation using [Qwen3Guard-Gen](https://huggingface.co/Qwen) models (0.6B, 4B, 8B).

## Structure

```
qwen3guard/
├── infer.py              # Run inference on all 3 model sizes
├── compare_schemas.py    # Evaluate model outputs against manual annotations
├── models/
│   └── qwen3guard.py     # Model loading and generation logic
└── results/              # Output comparison reports (.txt)
```

## Inference

Runs all three Qwen3Guard-Gen model sizes sequentially and writes results to a JSONL file.

```bash
python infer.py \
  --model qwen3guard \
  --in-file path/to/input.tsv \
  --out-file path/to/output.jsonl
```

### Input format

A TSV file with the following columns:

| Column       | Required | Description                                      |
|--------------|----------|--------------------------------------------------|
| `modality`   | yes      | `text` or `speech`                               |
| `user_text`  | yes      | The user's text input                            |
| `output`     | yes      | The model's response to evaluate                 |
| `transcript` | speech only | ASR transcript (required when modality=speech) |

### Output format

JSONL file with all original TSV fields plus, for each model size:

- `qwen3guard_{size}_Safety` — `Safe`, `Unsafe`, or `Controversial`
- `qwen3guard_{size}_Refusal` — `Yes` or `No`

## Evaluation

Compares model outputs against manual annotations using three label schemas:

| Schema       | Description                                      |
|--------------|--------------------------------------------------|
| `Soft`       | 3-class: Safe / Unsafe / Controversial           |
| `Hard_loose` | 2-class: Safe+Controversial vs Unsafe            |
| `Hard_strict`| 2-class: Safe vs Controversial+Unsafe            |

```bash
python compare_schemas.py <lang>
```

Where `<lang>` is a 2-letter language code (e.g. `en`, `fr`, `de`). Reads from `outputs/<lang>.jsonl` and saves the report to `results/<lang>_compare_schemas.txt`.