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
import csv
import logging
import importlib
import json
import sys
import torch
from tqdm import tqdm
from typing import Any, Dict

from models.qwen3guard import QWEN3GUARD_MODELS


logger = logging.getLogger(__name__)

MODEL_MODULES = {
    "qwen3guard": "models.qwen3guard",
}

MODELS = sorted(list(MODEL_MODULES.keys()))


def setup_model(model_name):
    if model_name not in MODEL_MODULES:
        raise NotImplementedError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {', '.join(MODELS)}"
        )

    logger.info("Setting transformers seed to 42 for reproducibility.")
    try:
        from transformers.trainer_utils import set_seed
    except ImportError:
        from transformers import set_seed
    set_seed(42)

    module_name = MODEL_MODULES[model_name]
    logger.info(f"Importing module '{module_name}'")
    module = importlib.import_module(module_name)

    load_func = getattr(module, "load_model", None)
    if not load_func:
        raise ImportError(f"Module {module_name} does not define `load_model`")

    generate_func = getattr(module, "generate", None)
    if not generate_func:
        raise ImportError(f"Module {module_name} does not define `generate`")

    return load_func, generate_func


def construct_evaluation_input(row: Dict[str, Any]) -> str:
    """
    Construct the input string for evaluation based on modality.

    Text modality: uses user_text only
    Speech modality: concatenates transcript and user_text
    """
    modality = row.get('modality', '')

    if modality == 'speech':
        transcript = row.get('transcript', '').strip()
        user_text = row.get('user_text', '').strip()
        if not transcript:
            raise ValueError("transcript required for speech modality")
        return f"{transcript} {user_text}".strip()
    else:
        return row.get('user_text', '').strip()


def load_models_outputs_tsv(path) -> list[Dict[str, Any]]:
    """Read a TSV file and return a list of sample dicts.

    Each sample contains all original TSV fields plus a 'prompt' key
    constructed via construct_evaluation_input.
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample = dict(row)
            sample["prompt"] = construct_evaluation_input(row)
            samples.append(sample)
    return samples


def infer(args):

    logger.info(f"Reading input file: {args.in_file}")
    all_samples = load_models_outputs_tsv(args.in_file)
    logger.info(f"Loaded {len(all_samples)} samples")

    results = [dict(sample) for sample in all_samples]

    load_func, generate_func = setup_model(args.model)

    n_models = len(QWEN3GUARD_MODELS)
    for model_idx, model_name in enumerate(QWEN3GUARD_MODELS, start=1):
        size_tag = model_name.split("-")[-1]
        safety_key = f"qwen3guard_{size_tag}_Safety"
        refusal_key = f"qwen3guard_{size_tag}_Refusal"

        logger.info(f"[{model_idx}/{n_models}] Loading {model_name} ...")
        model, tokenizer = load_func(model_name)
        logger.info(f"[{model_idx}/{n_models}] {model_name} loaded — running inference on {len(all_samples)} samples")
        logger.info(f"Model device: {model.device}")
        logger.info(f"Model dtype: {model.dtype}")

        for i, sample in enumerate(tqdm(all_samples, desc=model_name, unit="sample")):
            safe_label, _category_label, refusal_label = generate_func(model, tokenizer, sample)
            results[i][safety_key] = safe_label
            results[i][refusal_key] = refusal_label

        logger.info(f"[{model_idx}/{n_models}] Done with {model_name} — freeing GPU memory")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = args.out_file or "stdout"
    logger.info(f"Writing {len(results)} results to {out_path}")
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as outfile:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        for result in results:
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info("Inference complete.")


def add_infer_args(parser):
    parser.add_argument(
        "--model", required=True,
        help="Model to use for inference. Supported: " + ", ".join(MODELS),
    )
    parser.add_argument(
        "--in-file", required=True,
        help="Input TSV file with columns: modality, user_text, output, and optionally transcript.",
    )
    parser.add_argument(
        "--out-file", required=False, default=None,
        help="Output JSONL file path. If not set: stdout.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speech Safety - Qwen3Guard Inference"
    )
    parser = add_infer_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    infer(args)