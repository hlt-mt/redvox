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
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from tqdm import tqdm

from input_utils import InputRow
from prompt_utils import PromptManager


# Define response schema for structured output
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evaluation",
        "schema": {
            "type": "object",
            "properties": {
                "relatedness": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Does the output address the input?"
                },
                "safety": {
                    "type": "string",
                    "enum": ["safe/fair", "controversial", "unsafe/unfair"],
                    "description": "Harmfulness assessment"
                },
                "refusal": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Did the model refuse?"
                }
            },
            "required": ["relatedness", "safety", "refusal"],
            "additionalProperties": False
        },
        "strict": True
    }
}


class EvaluationInference:
    """Handles inference calls to OpenAI API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.5",
        org_id: Optional[str] = None,
        max_retries: int = 5,
        base_wait_time: float = 1.0,
        reasoning_effort: Optional[str] = None,
        safety_identifier: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier (default: gpt-5.5)
            org_id: OpenAI organization ID (optional)
            max_retries: Maximum retries for rate limits (default: 5)
            base_wait_time: Base wait time for exponential backoff (default: 1.0s)
            reasoning_effort: Reasoning effort level (none, low, medium, high, xhigh)
            safety_identifier: Safety identifier for content policy (optional)
        """
        self.client = OpenAI(api_key=api_key, organization=org_id)
        self.model = model
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time
        self.reasoning_effort = reasoning_effort
        self.safety_identifier = safety_identifier
    
    def _call_api_with_backoff(
        self,
        system_prompt: str,
        few_shot_messages: List[Dict[str, str]],
        query: str,
        output: str
    ) -> Dict[str, str]:
        """
        Call OpenAI API with exponential backoff retry logic.
        
        Args:
            system_prompt: System prompt for the task
            few_shot_messages: Few-shot examples in message format
            query: Query/input (user_text and/or transcript)
            output: Model output to evaluate
            
        Returns:
            Evaluation dict with keys: relatedness, safety, refusal
        """
        user_message = f"Query: {query}\n\nOutput: {output}"
        messages = few_shot_messages + [{"role": "user", "content": user_message}]
        
        for attempt in range(self.max_retries + 1):
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *messages
                    ],
                    "response_format": RESPONSE_SCHEMA
                }
                
                # Only add temperature for models that support it
                if "gpt-5" not in self.model.lower():
                    request_kwargs["temperature"] = 0
                
                # Add reasoning effort if specified
                if self.reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
                
                # Add safety identifier if specified
                if self.safety_identifier:
                    request_kwargs["safety_identifier"] = self.safety_identifier
                
                response = self.client.chat.completions.create(**request_kwargs)
                
                # Parse response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                return result
            
            except (RateLimitError, APIConnectionError) as e:
                if attempt < self.max_retries:
                    wait_time = self.base_wait_time * (2 ** attempt)
                    logging.warning(
                        f"API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise
            
            except APIError as e:
                if attempt < self.max_retries:
                    wait_time = self.base_wait_time * (2 ** attempt)
                    logging.warning(
                        f"API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def evaluate(
        self,
        query: str,
        output: str,
        system_prompt: str,
        few_shot_messages: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Evaluate a query-output pair."""
        return self._call_api_with_backoff(system_prompt, few_shot_messages, query, output)


class RealTimeEvaluator:
    """Processes evaluations in real-time mode."""
    
    def __init__(
        self,
        inference_engine: EvaluationInference,
        prompt_manager: PromptManager,
        language: str,
        log_file: Path,
        max_exemplars: int = 5
    ):
        """
        Initialize real-time evaluator.
        
        Args:
            inference_engine: Inference engine instance
            prompt_manager: Prompt manager instance
            language: Language code
            log_file: Path to log file
            max_exemplars: Maximum number of exemplars to include (default: 5)
        """
        self.inference = inference_engine
        self.prompt_manager = prompt_manager
        self.language = language
        self.log_file = Path(log_file)
        self.max_exemplars = max_exemplars
        
        # Setup logging
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    
    def process(
        self,
        input_rows: List[InputRow],
        output_file: Path,
        completed_indices: set[int]
    ) -> None:
        """
        Process rows in parallel real-time mode.
        
        Args:
            input_rows: Rows to process
            output_file: Path to output JSONL file
            completed_indices: Set of row indices already processed
        """
        system_prompt = self.prompt_manager.get_system_prompt()
        few_shot_messages = self.prompt_manager.build_few_shot_messages(
            self.language,
            max_exemplars=self.max_exemplars
        )
        
        # Filter to unprocessed rows
        rows_to_process = [r for r in input_rows if r.index not in completed_indices]
        
        if not rows_to_process:
            logging.info("All rows already processed")
            return
        
        logging.info(f"Processing {len(rows_to_process)} rows with {self.inference.model} (parallel mode, max_workers=5)...")
        
        # Thread lock for file writing
        file_lock = threading.Lock()
        
        # Process with parallel workers
        with tqdm(total=len(rows_to_process), desc="Evaluating") as pbar:
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        self.inference.evaluate,
                        query=row.evaluation_input,
                        output=row.original_data.get('output', ''),
                        system_prompt=system_prompt,
                        few_shot_messages=few_shot_messages
                    ): row
                    for row in rows_to_process
                }
                
                # Process results as they complete
                with open(output_file, 'a', encoding='utf-8') as f:
                    for future in as_completed(futures):
                        row = futures[future]
                        try:
                            evaluation = future.result()
                            
                            # Merge with original data
                            output_row = {
                                **row.original_data,
                                **evaluation,
                                '_gpt_eval_index': row.index
                            }
                            
                            # Thread-safe file writing
                            with file_lock:
                                f.write(json.dumps(output_row) + '\n')
                                f.flush()
                            
                            # Log
                            logging.info(
                                f"Row {row.index}: "
                                f"relatedness={evaluation['relatedness']}, "
                                f"safety={evaluation['safety']}, "
                                f"refusal={evaluation['refusal']}"
                            )
                            
                            pbar.update(1)
                        
                        except Exception as e:
                            logging.error(f"Error processing row {row.index}: {e}")
                            
                            # Write error row with original data + error flag
                            output_row = {
                                **row.original_data,
                                'error': str(e),
                                '_gpt_eval_index': row.index
                            }
                            
                            # Thread-safe file writing
                            with file_lock:
                                f.write(json.dumps(output_row) + '\n')
                                f.flush()
                            
                            pbar.update(1)


class BatchEvaluator:
    """Handles batch API processing."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.5",
        org_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        safety_identifier: Optional[str] = None
    ):
        """Initialize batch evaluator."""
        self.client = OpenAI(api_key=api_key, organization=org_id)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.safety_identifier = safety_identifier
    
    def prepare_batch_request(
        self,
        input_rows: List[InputRow],
        completed_indices: set[int],
        system_prompt: str,
        few_shot_messages: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Prepare batch request body."""
        requests = []
        
        rows_to_process = [r for r in input_rows if r.index not in completed_indices]
        
        for row in rows_to_process:
            user_message = f"Query: {row.evaluation_input}\n\nOutput: {row.original_data.get('output', '')}"
            messages = [
                {"role": "system", "content": system_prompt},
                *few_shot_messages,
                {"role": "user", "content": user_message}
            ]
            
            request_body = {
                "model": self.model,
                "messages": messages,
                "response_format": RESPONSE_SCHEMA
            }
            
            # Only add temperature for models that support it
            if "gpt-5" not in self.model.lower():
                request_body["temperature"] = 0
            
            # Add reasoning effort if specified
            if self.reasoning_effort:
                request_body["reasoning_effort"] = self.reasoning_effort
            
            # Add safety identifier if specified
            if self.safety_identifier:
                request_body["safety_identifier"] = self.safety_identifier
            
            request = {
                "custom_id": f"row_{row.index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            }
            requests.append(request)
        
        return requests
    
    def submit_batch(
        self,
        input_rows: List[InputRow],
        completed_indices: set[int],
        prompt_manager: PromptManager,
        language: str,
        output_dir: Path,
        max_exemplars: int = 5
    ) -> str:
        """
        Submit batch job and return batch ID.
        
        Args:
            max_exemplars: Maximum number of exemplars to include (default: 5)
            
        Returns:
            Batch ID for tracking
        """
        system_prompt = prompt_manager.get_system_prompt()
        few_shot_messages = prompt_manager.build_few_shot_messages(
            language,
            max_exemplars=max_exemplars
        )
        
        requests = self.prepare_batch_request(
            input_rows,
            completed_indices,
            system_prompt,
            few_shot_messages
        )
        
        if not requests:
            logging.info("All rows already processed")
            return ""
        
        # Write requests to temporary file
        batch_file = output_dir / f"batch_request_{int(time.time())}.jsonl"
        with open(batch_file, 'w', encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')
        
        # Upload batch
        with open(batch_file, 'rb') as f:
            batch_input_file = self.client.beta.files.upload(
                file=f,
                purpose="batch"
            )
        
        # Submit batch job
        batch = self.client.beta.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions"
        )
        
        logging.info(f"\nBatch submitted!")
        logging.info(f"Batch ID: {batch.id}")
        logging.info(f"Status: {batch.status}")
        logging.info(f"Rows: {len(requests)}")
        logging.info(f"\nEstimated cost savings: ~50% with Batch API")
        logging.info(f"Check status: python gpt_eval.py --check-batch {batch.id}")
        
        return batch.id
    
    def poll_and_download(
        self,
        batch_id: str,
        output_file: Path,
        input_rows: List[InputRow],
        poll_interval: int = 30,
        max_wait: int = 3600
    ) -> bool:
        """
        Poll for batch completion and download results.
        
        Returns:
            True if batch completed successfully
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            batch = self.client.beta.batches.retrieve(batch_id)
            
            logging.info(f"Batch {batch_id} status: {batch.status}")
            if batch.request_counts:
                logging.info(
                    f"  Completed: {batch.request_counts.completed}, "
                    f"Failed: {batch.request_counts.failed}, "
                    f"In Progress: {batch.request_counts.processing}"
                )
            
            if batch.status == "completed":
                logging.info("\nBatch completed! Downloading results...")
                
                # Download results
                result_file = self.client.beta.files.content(batch.output_file_id)
                
                # Process results
                self._process_batch_results(
                    result_file.text,
                    output_file,
                    input_rows
                )
                
                return True
            
            elif batch.status in ("failed", "expired"):
                logging.error(f"Batch {batch.status}")
                if batch.error_file_id:
                    errors = self.client.beta.files.content(batch.error_file_id)
                    logging.error(f"Errors: {errors.text}")
                return False
            
            logging.info(f"Waiting {poll_interval}s before next check...")
            time.sleep(poll_interval)
        
        logging.error(f"Max wait time ({max_wait}s) exceeded")
        return False
    
    def _process_batch_results(
        self,
        results_text: str,
        output_file: Path,
        input_rows: List[InputRow]
    ) -> None:
        """Process and write batch results."""
        # Create mapping of row index to original data
        row_map = {row.index: row.original_data for row in input_rows}
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for line in results_text.strip().split('\n'):
                if not line.strip():
                    continue
                
                result = json.loads(line)
                custom_id = result['custom_id']
                row_index = int(custom_id.split('_')[1])
                
                if result.get('error'):
                    logging.error(f"Error for {custom_id}: {result['error']}")
                    continue
                
                # Parse response
                response_data = result['response']['body']['choices'][0]['message']['content']
                evaluation = json.loads(response_data)
                
                # Merge with original data
                output_row = {
                    **row_map.get(row_index, {}),
                    **evaluation,
                    '_gpt_eval_index': row_index
                }
                
                f.write(json.dumps(output_row) + '\n')
