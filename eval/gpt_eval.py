#!/usr/bin/env python3

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

import argparse
import os
import sys
import logging
from pathlib import Path

from input_utils import parse_input_rows, load_completed_indices
from prompt_utils import PromptManager
from inference import (
    EvaluationInference,
    RealTimeEvaluator,
    BatchEvaluator
)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add command line arguments to parser."""
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Path to input .jsonl or .tsv file'
    )
    parser.add_argument(
        '-l', '--lang',
        type=str,
        required=True,
        choices=['en', 'es', 'fr', 'de', 'it'],
        help='Language code (BCP-47)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Path to output .jsonl file'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gpt-5.5',
        help='OpenAI model identifier (default: gpt-5.5)'
    )
    parser.add_argument(
        '--max-exemplars',
        type=int,
        default=5,
        help='Maximum number of exemplars to include in few-shot prompts (default: 5)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Use OpenAI Batch API instead of real-time inference'
    )
    parser.add_argument(
        '--check-batch',
        type=str,
        default=None,
        help='Check status of a batch job (provide batch ID)'
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        default=None,
        choices=['none', 'low', 'medium', 'high', 'xhigh'],
        help='Reasoning effort level (default: model default, typically medium)'
    )
    parser.add_argument(
        '--safety-identifier',
        type=str,
        default=None,
        help='Safety identifier for content policy (optional)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation tool using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time evaluation (requires OPENAI_API_KEY env var)
  python gpt_eval.py -i input.tsv -l en -o output.jsonl

  # Batch evaluation
  python gpt_eval.py -i input.tsv -l en -o output.jsonl --batch

  # With custom model
  python gpt_eval.py -i input.tsv -l en -o output.jsonl -m gpt-5.5
"""
    )
    parser = add_args(parser)
    args = parser.parse_args()
    
    # Get API credentials from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    org_id = os.getenv('OPENAI_ORG_ID')
    
    # Handle batch status check
    if args.check_batch:
        batch_evaluator = BatchEvaluator(api_key, args.model, org_id)
        batch = batch_evaluator.client.beta.batches.retrieve(args.check_batch)
        logging.info(f"Batch {args.check_batch} status: {batch.status}")
        if batch.request_counts:
            logging.info(
                f"  Completed: {batch.request_counts.completed}, "
                f"Failed: {batch.request_counts.failed}, "
                f"In Progress: {batch.request_counts.processing}"
            )
        return
    
    # Validate arguments (creates output directory)
    validate_arguments(args)
    
    # Setup logging file
    log_file = args.output.with_suffix(args.output.suffix + '.log')
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    try:
        # Load input
        logging.info(f"Loading input from {args.input}...")
        input_rows = parse_input_rows(args.input, args.lang)
        logging.info(f"Loaded {len(input_rows)} rows")
        
        # Check for completed rows
        completed_indices = load_completed_indices(args.output)
        if completed_indices:
            logging.info(f"Found {len(completed_indices)} already processed rows")
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        
        if args.batch:
            # Batch processing
            logging.info("Submitting batch job...")
            batch_evaluator = BatchEvaluator(
                api_key,
                args.model,
                org_id,
                reasoning_effort=args.reasoning_effort,
                safety_identifier=args.safety_identifier
            )
            
            batch_id = batch_evaluator.submit_batch(
                input_rows,
                completed_indices,
                prompt_manager,
                args.lang,
                args.output.parent,
                max_exemplars=args.max_exemplars
            )
            
            if batch_id:
                logging.info("Polling for batch completion...")
                success = batch_evaluator.poll_and_download(
                    batch_id,
                    args.output,
                    input_rows
                )
                
                if success:
                    logging.info(f"Results written to {args.output}")
                    logging.info(f"Log written to {log_file}")
        else:
            # Real-time processing
            inference_engine = EvaluationInference(
                api_key=api_key,
                model=args.model,
                org_id=org_id,
                reasoning_effort=args.reasoning_effort,
                safety_identifier=args.safety_identifier
            )
            
            evaluator = RealTimeEvaluator(
                inference_engine,
                prompt_manager,
                args.lang,
                log_file,
                max_exemplars=args.max_exemplars
            )
            
            evaluator.process(input_rows, args.output, completed_indices)
            logging.info(f"Results written to {args.output}")
            logging.info(f"Log written to {log_file}")
    
    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Progress saved to output file.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
