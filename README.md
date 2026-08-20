# RedVox

<p align="center">
<img src="https://github.com/hlt-mt/redvox/blob/main/artifacts/logo.png?raw=true" width="350"/>
</p>

This repository contains the **code, model outputs, and evaluation resources** for the paper:

> **RedVox: Safety and Fairness Gaps in Speech Models Across Languages**
> Beatrice Savoldi*, Sara Papi*, Wafa Aissa, Matteo Negri, and Luisa Bentivogli, 2026.

[📄 Paper](https://arxiv.org/abs/2606.26968) · [🤗 Dataset](https://huggingface.co/datasets/hlt-mt/RedVox) · [📦 Code & Evaluation](https://github.com/hlt-mt/redvox)

## Overview

RedVox is a multilingual red teaming benchmark designed to study **safety and fairness in speech and audio models across languages**.

The benchmark evaluates how models respond to harmful requests and stereotypical content when the input is provided through different audio conditions. It covers **English, German, Spanish, French, and Italian**, and includes both human speech and controlled audio variants with silence and background noise.

The benchmark is used to investigate safety and fairness behavior across:

* **Languages:** English, German, Spanish, French, and Italian
* **Modalities:** speech and audio
* **Audio conditions:** speech, silence, and two background-noise variants
* **Vulnerability types:** stereotypes and unsafe requests
* **Model families:** speech and multimodal models, including both open and API-based systems

## Repository Structure

```text
.
├── artifacts/       # Guidelines and supporting artifacts (e.g., logo)
├── eval/            # Evaluation data and scripts for LLM-as-a-judge. See [`eval/README.md`](eval/README.md) for details.
├── models/          # Model implementations
├── outputs/         # Generated model outputs and manual evaluation
├── qwen3guard/      # Qwen3Guard inference and evaluation
├── scripts/         # Utility scripts for data processing and annotation
├── infer.py         # Main inference entry point
├── LICENSE
└── README.md
```

## Dataset

The RedVox test set contains **3,414 samples** across five languages:

| Language  |   Samples |
| --------- | --------: |
| English   |     1,359 |
| German    |       519 |
| Spanish   |       354 |
| French    |       401 |
| Italian   |       781 |
| **Total** | **3,414** |

Each sample is associated with an audio input and metadata describing the language, vulnerability type, audio condition, and other relevant information.

The dataset contains:

* **Speech:** original harmful requests spoken by human participants.
* **Silence:** speech-related content replaced with background silence.
* **Noise A:** speech with background noise variant A.
* **Noise B:** speech with background noise variant B.

The dataset is released on Hugging Face:

**[🤗 RedVox Dataset](https://huggingface.co/datasets/hlt-mt/RedVox)**

Please refer to the dataset card for the complete metadata schema, statistics, and usage information.

## Inference

The repository provides a single inference entry point implemented in [infer.py](infer.py#L1-L200):

```bash
python infer.py --model <MODEL> --language <LANG> [--out-file PATH] [--audio-dir DIR] [--text-only] [--continue]
```

- **Models:** Implementations live under [`models/`](models/). Supported model names include: `gemma4`, `qwen3omni`, `phi4multimodal`, `voxtral`, `qwen2audio`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, and `gpt-realtime2` (see `--model` help).
- **Languages:** `--language` is required and must be one of `en`, `de`, `es`, `fr`, or `it` — the script loads the `test` split of the HuggingFace dataset `FBK-MT/redvox` for that language.
- **Audio files:** By default the dataset's `audio` path is used as-is. Supply `--audio-dir` to resolve audio file names relative to a local directory.
- **Text-only mode:** Use `--text-only` to run inference on the text-only part of the dataset.
- **Output:** The script writes one JSONL object per sample containing the original dataset fields plus an `output` string with the model response. Use `--out-file` to write to a file; otherwise output is written to `stdout`.
- **Continue:** `--continue` appends new outputs to an existing `--out-file` and skips already-processed samples (requires `--out-file`).

Examples:

```bash
python infer.py --model gemma4 --language en --out-file outputs/en/gemma4.jsonl --audio-dir /data/redvox/audio

python infer.py --model qwen2audio --language es --text-only --out-file outputs/text-only/es/qwen2audio.jsonl

python infer.py --model gemma4 --language en --out-file outputs/en/gemma4.jsonl --continue
```

Some models require extra dependencies, model weights, or API credentials.

## Automatic Evaluation

Model outputs are provided in [`outputs/`](outputs/), while evaluation resources (LLM-as-a-judge) are available in [`eval/`](eval/). The additional resources for Qwen3Guard judge are under `qwen3guard/`](qwen3guard/).


### Manual Evaluation

A subset of the benchmark was manually evaluated using sentence-level judge labels. The corresponding data and guidelines are available under [`outputs/manual_eval`](outputs/manual_eval) and [`artifacts/`](artifacts/), respectively.


## Citation

If you use RedVox, its evaluation resources, or the accompanying code in your research, please cite:

```bibtex
@misc{savoldi2026redvoxsafetyfairnessgaps,
      title={RedVox: Safety and Fairness Gaps in Speech Models Across Languages},
      author={Beatrice Savoldi and Sara Papi and Wafa Aissa and Matteo Negri and Luisa Bentivogli},
      year={2026},
      eprint={2606.26968},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.26968}
}
```

## License

The code is released under the Apache 2.0 license (see [`LICENSE`](LICENSE)).

The RedVox dataset is distributed separately. Please refer to the [dataset card](https://huggingface.co/datasets/hlt-mt/RedVox) for its licensing and usage terms.
