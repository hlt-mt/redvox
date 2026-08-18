# Safety Evaluation with GPT Judge

Safety evaluation of LLM outputs using OpenAI GPT-5.5 as judge.

## Structure

```
eval/
├── gpt_eval.py           # Main evaluation script
├── inference.py          # OpenAI API inference
├── input_utils.py        # Input parsing and validation
├── prompt_utils.py       # Prompt construction with language variants
├── requirements.txt      # Python dependencies
├── prompts/
│   ├── system_prompt.txt # System prompt for safety evaluation
│   └── exemplars/        # Few-shot examples by language
└── __init__.py
```

## Evaluation

Run GPT-based evaluation on model outputs.

```bash
python gpt_eval.py \
  --input path/to/input.jsonl \
  --lang <lang> \
  --model <model> \
  --output path/to/output.jsonl \
  [--mode realtime|batch] \
  [--reasoning-effort high|medium|low]
```

**Required arguments:**
- `--input` — Path to input JSONL file
- `--lang` — Language code (en, fr, de, es, it)
- `--model` — `gpt-5.5-2026-04-23`
- `--output` — Path to output JSONL file

**Optional arguments:**
- `--mode` — Evaluation mode (default: `realtime`)
- `--reasoning-effort` — (default: model default, `medium`)

### Input format

A JSONL file with the following fields:

| Field        | Required | Description                                      |
|--------------|----------|--------------------------------------------------|
| `id`         | yes      | Unique identifier                                |
| `language`   | yes      | Language code (en, fr, de, es, it)               |
| `modality`   | yes      | `text` or `speech`                               |
| `user_text`  | yes      | The user's input                                 |
| `output`     | yes      | Model response to evaluate                       |
| `transcript` | speech only | ASR transcript (required when modality=speech) |

### Output format

JSONL file with all input fields plus GPT judgments:

| Field | Required | Description |
|-------|----------|-------------|
| `safety` | yes | `safe/fair`, `controversial`, or `unsafe/unfair` |
| `relatedness` | yes | `yes` or `no` |
