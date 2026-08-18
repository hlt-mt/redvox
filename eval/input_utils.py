# Copyright 2026 FBK
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Literal
from dataclasses import dataclass

from pydantic import BaseModel, field_validator


class EvaluationInput(BaseModel):
    """Validated input structure for evaluation."""
    lang: str
    modality: Literal["text", "speech"]
    user_text: str
    output: str
    transcript: str | None = None
    
    @field_validator('modality')
    @classmethod
    def validate_modality(cls, v: str) -> str:
        if v not in ("text", "speech"):
            raise ValueError("modality must be 'text' or 'speech'")
        return v
    
    @field_validator('transcript', mode='after')
    @classmethod
    def validate_transcript(cls, v: str | None, info) -> str | None:
        if info.data.get('modality') == 'speech' and not v:
            raise ValueError("transcript is required when modality is 'speech'")
        return v


@dataclass
class InputRow:
    """A single row from input file with constructed evaluation input."""
    index: int
    original_data: Dict[str, Any]
    evaluation_input: str  # The constructed input for LLM
    
    @property
    def unique_id(self) -> str:
        """Unique identifier for resuming."""
        return f"row_{self.index}"


def construct_evaluation_input(row: Dict[str, Any]) -> str:
    """
    Construct the input string for evaluation based on modality.
    
    Text modality: uses user_text only
    Speech modality: concatenates transcript and user_text
    """
    modality = row.get('modality', '').lower()
    
    if modality == 'speech':
        transcript = row.get('transcript', '').strip()
        user_text = row.get('user_text', '').strip()
        if not transcript:
            raise ValueError("transcript required for speech modality")
        return f"{transcript} {user_text}".strip()
    else:
        return row.get('user_text', '').strip()


def load_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tsv_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a TSV file."""
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(dict(row))
    return rows


def load_input_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load input file (JSONL or TSV)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if file_path.suffix.lower() == '.jsonl':
        return load_jsonl_file(file_path)
    elif file_path.suffix.lower() == '.tsv':
        return load_tsv_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .jsonl or .tsv")


def parse_input_rows(
    file_path: Path,
    language: str
) -> List[InputRow]:
    """
    Parse input file and construct evaluation rows.
    
    Args:
        file_path: Path to input JSONL/TSV file
        language: BCP-47 language code to validate against
        
    Returns:
        List of InputRow objects with validated data
    """
    rows_data = load_input_file(file_path)
    
    input_rows = []
    for idx, row in enumerate(rows_data):
        try:
            # Validate required fields
            EvaluationInput(
                lang=row.get('lang', language),
                modality=row.get('modality', 'text'),
                user_text=row.get('user_text', ''),
                output=row.get('output', ''),
                transcript=row.get('transcript')
            )
            
            # Construct evaluation input
            eval_input = construct_evaluation_input(row)
            
            input_rows.append(InputRow(
                index=idx,
                original_data=row,
                evaluation_input=eval_input
            ))
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid row {idx}: {e}") from e
    
    return input_rows


def load_completed_indices(output_file: Path) -> set[int]:
    """Load indices of already-processed rows from output file."""
    if not output_file.exists():
        return set()
    
    completed = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Check if row has evaluation fields
                    if all(k in data for k in ['relatedness', 'safety', 'refusal']):
                        # Try to extract index from unique_id if present
                        if '_gpt_eval_index' in data:
                            completed.add(data['_gpt_eval_index'])
                except json.JSONDecodeError:
                    # Skip malformed JSON lines; continue to next line
                    continue
    
    return completed
