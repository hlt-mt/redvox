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
from pathlib import Path
from typing import List, Dict, Any


class PromptManager:
    """Manages system prompts and language-specific exemplars."""
    
    def __init__(self, prompts_dir: Path | None = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_dir: Path to prompts directory. If None, uses 'prompts' relative to script.
                Contains:
                - system_prompt.txt
                - exemplars/{language}.jsonl
        """
        if prompts_dir is None:
            # Use prompts directory relative to this script
            prompts_dir = Path(__file__).parent / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._validate_structure()
        self._system_prompt = None
        self._exemplars_cache = {}
    
    def _validate_structure(self) -> None:
        """Validate that required prompt files exist."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {self.prompts_dir}\n"
                "Create the following structure:\n"
                "prompts/\n"
                "  system_prompt.txt\n"
                "  exemplars/\n"
                "    en.jsonl\n"
                "    es.jsonl\n"
                "    fr.jsonl\n"
                "    de.jsonl\n"
                "    it.jsonl"
            )
        
        system_prompt_file = self.prompts_dir / "system_prompt.txt"
        if not system_prompt_file.exists():
            raise FileNotFoundError(f"system_prompt.txt not found in {self.prompts_dir}")
        
        exemplars_dir = self.prompts_dir / "exemplars"
        if not exemplars_dir.exists():
            raise FileNotFoundError(f"exemplars directory not found in {self.prompts_dir}")
    
    def get_system_prompt(self) -> str:
        """Load and cache system prompt."""
        if self._system_prompt is None:
            system_prompt_file = self.prompts_dir / "system_prompt.txt"
            with open(system_prompt_file, 'r', encoding='utf-8') as f:
                self._system_prompt = f.read().strip()
        return self._system_prompt
    
    def get_exemplars(self, language: str) -> List[Dict[str, Any]]:
        """
        Load language-specific exemplars.
        
        Args:
            language: BCP-47 language code (e.g., 'en', 'es', 'fr', 'de', 'it')
            
        Returns:
            List of exemplar dicts with keys: query, output, relatedness, safety, refusal
        """
        if language in self._exemplars_cache:
            return self._exemplars_cache[language]
        
        exemplar_file = self.prompts_dir / "exemplars" / f"{language}.jsonl"
        if not exemplar_file.exists():
            raise FileNotFoundError(
                f"Exemplar file not found: {exemplar_file}\n"
                f"Supported languages: en, es, fr, de, it"
            )
        
        exemplars = []
        with open(exemplar_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    exemplars.append(json.loads(line))
        
        self._exemplars_cache[language] = exemplars
        return exemplars
    
    def build_few_shot_messages(
        self,
        language: str,
        max_exemplars: int = 5
    ) -> List[Dict[str, str]]:
        """
        Build few-shot messages for chat completions API.
        
        Args:
            language: Language code for exemplar selection
            max_exemplars: Maximum number of exemplars to include
            
        Returns:
            List of message dicts with role and content for few-shot prompt
        """
        exemplars = self.get_exemplars(language)
        
        # Use up to max_exemplars
        selected_exemplars = exemplars[:max_exemplars]
        
        messages = []
        
        for exemplar in selected_exemplars:
            # User message with query and output to evaluate
            user_content = f"Query: {exemplar.get('query', '')}\n\nOutput: {exemplar.get('output', '')}"
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Assistant message with judgment
            judgment = {
                "relatedness": exemplar.get("relatedness"),
                "safety": exemplar.get("safety"),
                "refusal": exemplar.get("refusal")
            }
            messages.append({
                "role": "assistant",
                "content": json.dumps(judgment)
            })
        
        return messages

