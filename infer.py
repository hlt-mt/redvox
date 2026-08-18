# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import logging
import importlib
import json
import os
from tqdm import tqdm
import sys


MODEL_MODULES = {
    # omni
    "gemma4": "models.omni.gemma4",
    "qwen3omni": "models.omni.qwen3omni",
    "phi4multimodal": "models.omni.phi4multimodal",

    # speechllm
    "voxtral": "models.speechllm.voxtral",
    "qwen2audio": "models.speechllm.qwen2audio",

    # proprietary
    "gemini-3.1-flash-lite": "models.api.gemini",
    "gemini-3.1-pro-preview": "models.api.gemini",
    "gpt-realtime2": "models.api.gpt-realtime2",
}

MODELS = sorted(list(MODEL_MODULES.keys()))

# Models that don't use torch — skip torch seed setting (it's slow)
NON_TORCH_MODELS = ["gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gpt-realtime2"]


def setup_model(model_name):
    if model_name not in MODEL_MODULES:
        raise NotImplementedError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {', '.join(MODELS)}"
        )

    if model_name not in NON_TORCH_MODELS:
        logging.info("Setting transformers seed to 42 for reproducibility.")
        try:
            from transformers.trainer_utils import set_seed
        except ImportError:
            from transformers import set_seed
        set_seed(42)
    else:
        logging.info(
            f"Skipping set_seed because '{model_name}' is a non-torch model."
        )

    module_name = MODEL_MODULES[model_name]
    module = importlib.import_module(module_name)

    load_func = getattr(module, "load_model", None)
    if not load_func:
        raise ImportError(f"Module {module_name} does not define `load_model`")

    generate_func = getattr(module, "generate", None)
    if not generate_func:
        raise ImportError(f"Module {module_name} does not define `generate`")

    model = load_func(model_name)
    return model, generate_func


def load_hf_dataset(dataset_name: str, language: str):
    """
    Load the HuggingFace dataset and yield samples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required. Install it with: pip install datasets"
        )

    assert language in ["en", "de", "es", "fr", "it"]

    logging.info(f"Loading HuggingFace dataset '{dataset_name}'")
    ds = load_dataset(dataset_name, language, split="test")

    logging.info(f"Dataset columns: {ds.column_names}")

    for i in range(len(ds)):
        yield {col: ds[col][i] for col in ds.column_names}


def infer(args):
    logging.info(f"Loading model {args.model}")
    model, generate = setup_model(args.model)

    all_samples = list(load_hf_dataset("FBK-MT/redvox", args.language))

    # Skip already-processed samples, if any
    if vars(args)["continue"]:
        if not args.out_file:
            raise ValueError("To use --continue, --out-file must be set.")
        if not os.path.exists(args.out_file):
            already_done = 0
        else:
            with open(args.out_file, "r") as f:
                already_done = sum(1 for _ in f)
        logging.info(f"Skipping {already_done} already processed samples.")
        all_samples = all_samples[already_done:]
        outfile = open(args.out_file, "a", encoding="utf-8")
    elif args.out_file:
        outfile = open(args.out_file, "w", encoding="utf-8")
    else:
        outfile = sys.stdout

    for sample in tqdm(all_samples, desc="Generating Outputs"):
        audio_path = (
            os.path.join(args.audio_dir, os.path.basename(sample["audio"]))
            if args.audio_dir
            else sample["audio"]
        )

        model_input = {
            "lang": sample["lang"],
            "prompt": sample["user_text"],
            "sample": audio_path,
            "text_only": args.text_only,
        }

        output = generate(model, model_input).strip()

        # Preserve all metadata fields and append the model output
        result = {**sample, "output": output}

        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.flush()

    if args.out_file:
        outfile.close()


def add_infer_args(parser):
    parser.add_argument(
        "--audio-dir", default=None,
        help=(
            "Directory where audio files are stored locally. "
            "If provided, file_name is resolved relative to this directory. "
            "If omitted, file_name is used as-is."
        ),
    )
    parser.add_argument(
        "--language", required=True,
        help="Language (2 code) to use for inference. "
    )
    parser.add_argument(
        "--model", required=True,
        help="Model to use for inference. Supported: " + ", ".join(MODELS),
    )
    parser.add_argument(
        "--out-file", required=False, default=None,
        help="Output JSONL file path. If not set: stdout.",
    )
    parser.add_argument(
        "--continue", default=False, action="store_true",
        help="Append new outputs to an existing --out-file, skipping already processed samples.",
    )
    parser.add_argument(
        "--text-only", default=False, action="store_true",
        help = (
            "Run inference using only the text prompt (user_text), "
            "without providing the audio sample to the model."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech Safety - Model Inference"
    )
    parser = add_infer_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    infer(args)