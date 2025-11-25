#!/usr/bin/env python3
"""
Parallel Multi-LLM Trace Collection Script
Runs multiple models and traces concurrently for faster collection

Usage:
    # Collect from all models in parallel (4-6 hours instead of 40-50)
    python collect_multi_llm_parallel.py --all --workers 7

    # Specific models with 3 concurrent traces per model
    python collect_multi_llm_parallel.py --model gpt51 claude45 --workers 2 --traces-per-worker 3
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.openrouter_llm import OpenRouterLLM
from agents.agent_wrapper import InstrumentedAgent
from agents.tasks import ALL_TASKS
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

# Load environment
load_dotenv()


# =============================================================================
# Mock Tools
# =============================================================================

def mock_search(query: str) -> str:
    return f"Search results for '{query}': [Simulated web results]"

def mock_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def mock_summarize(text: str) -> str:
    return f"Summary: {text[:100]}..."

def mock_code_writer(description: str) -> str:
    return f"# Generated code for: {description}\npass"

def mock_data_parser(data: str) -> str:
    return f"Parsed data: {data[:50]}..."

TOOLS = [
    StructuredTool.from_function(name="Search", func=mock_search, description="Search for information on the web"),
    StructuredTool.from_function(name="Calculator", func=mock_calculator, description="Perform mathematical calculations"),
    StructuredTool.from_function(name="Summarize", func=mock_summarize, description="Summarize text content"),
    StructuredTool.from_function(name="CodeWriter", func=mock_code_writer, description="Generate code based on description"),
    StructuredTool.from_function(name="DataParser", func=mock_data_parser, description="Parse and extract data from structured formats"),
]


# =============================================================================
# Worker Function (runs in separate process)
# =============================================================================

def collect_trace_worker(args):
    """Worker function to collect a single trace"""
    model_key, model_config, task, trace_type, trace_idx, config, output_base = args

    try:
        # Initialize LLM
        llm = OpenRouterLLM(
            model_id=model_config['model_id'],
            temperature=config['openrouter']['temperature'],
            max_tokens=config['openrouter']['max_tokens'],
            timeout=config['openrouter']['timeout_seconds']
        )

        # Create agent
        agent = InstrumentedAgent(llm=llm, tools=TOOLS)

        # Execute task
        trace = agent.execute(task.description)

        # Add metadata
        trace_dict = asdict(trace)
        trace_dict['metadata'] = {
            'model_key': model_key,
            'model_id': model_config['model_id'],
            'model_name': model_config['display_name'],
            'provider': model_config['provider'],
            'architecture': model_config['architecture'],
            'trace_type': trace_type,
            'task_category': task.category,
            'collected_at': datetime.utcnow().isoformat()
        }

        # Save trace
        output_dir = Path(output_base) / model_key / trace_type
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_file = output_dir / f"trace_{trace_idx:04d}.json"
        with open(trace_file, 'w') as f:
            json.dump(trace_dict, f, indent=2)

        return {
            'success': True,
            'model_key': model_key,
            'trace_type': trace_type,
            'trace_id': trace.trace_id,
            'duration_ms': trace.total_duration_ms
        }

    except Exception as e:
        return {
            'success': False,
            'model_key': model_key,
            'trace_type': trace_type,
            'error': str(e)
        }


# =============================================================================
# Progress Tracking (Shared across processes)
# =============================================================================

class SharedProgress:
    """Thread-safe progress tracking"""

    def __init__(self, manager, checkpoint_file):
        self.checkpoint_file = Path(checkpoint_file)
        self.progress = manager.dict()
        self.lock = manager.Lock()
        self.load()

    def load(self):
        """Load progress from checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                for key, value in data.get('models', {}).items():
                    self.progress[key] = value

    def save(self):
        """Save progress to checkpoint"""
        with self.lock:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'models': dict(self.progress),
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)

    def get_model_progress(self, model_key):
        """Get progress for a model"""
        if model_key not in self.progress:
            with self.lock:
                if model_key not in self.progress:
                    self.progress[model_key] = {
                        'benign_collected': 0,
                        'backdoor_collected': 0,
                        'total_collected': 0,
                        'errors': 0
                    }
        return dict(self.progress[model_key])

    def update(self, model_key, trace_type, success=True):
        """Update progress"""
        with self.lock:
            if model_key not in self.progress:
                self.progress[model_key] = {
                    'benign_collected': 0,
                    'backdoor_collected': 0,
                    'total_collected': 0,
                    'errors': 0
                }

            p = dict(self.progress[model_key])

            if success:
                if trace_type == 'benign':
                    p['benign_collected'] += 1
                else:
                    p['backdoor_collected'] += 1
                p['total_collected'] += 1
            else:
                p['errors'] += 1

            self.progress[model_key] = p
            self.save()


# =============================================================================
# Main Collection Function
# =============================================================================

def collect_parallel(config, models_to_collect, num_workers, traces_per_worker, benign_only, backdoor_only):
    """Collect traces in parallel across models and traces"""

    print(f"\n{'='*70}")
    print(f"🚀 Parallel Multi-LLM Trace Collection")
    print(f"{'='*70}")
    print(f"Models: {len(models_to_collect)}")
    print(f"Workers: {num_workers}")
    print(f"Traces per worker: {traces_per_worker}")
    print(f"Estimated parallelism: {num_workers * traces_per_worker}x")
    print(f"{'='*70}\n")

    # Setup shared progress
    manager = Manager()
    progress = SharedProgress(manager, config['progress']['checkpoint_file'])

    # Prepare all work items
    work_items = []
    output_base = config['output']['base_dir']

    for model_key in models_to_collect:
        model_config = config['models'][model_key]
        model_progress = progress.get_model_progress(model_key)

        # Benign traces
        if not backdoor_only:
            benign_needed = config['collection']['benign_traces_per_model'] - model_progress['benign_collected']
            for i in range(benign_needed):
                task = random.choice(ALL_TASKS)
                work_items.append((
                    model_key, model_config, task, 'benign',
                    model_progress['benign_collected'] + i, config, output_base
                ))

        # Backdoor traces
        if not benign_only:
            backdoor_needed = config['collection']['backdoor_traces_per_model'] - model_progress['backdoor_collected']
            for i in range(backdoor_needed):
                task = random.choice(ALL_TASKS)
                work_items.append((
                    model_key, model_config, task, 'backdoor',
                    model_progress['backdoor_collected'] + i, config, output_base
                ))

    # Shuffle for better load balancing
    random.shuffle(work_items)

    print(f"📊 Total work items: {len(work_items)}")
    print(f"⏱️  Estimated time: {len(work_items) / (num_workers * traces_per_worker) * 2:.0f}-{len(work_items) / (num_workers * traces_per_worker) * 3:.0f} minutes")
    print(f"\n🏃 Starting parallel collection...\n")

    # Execute in parallel
    start_time = time.time()
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=num_workers * traces_per_worker) as executor:
        futures = {executor.submit(collect_trace_worker, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()

            if result['success']:
                completed += 1
                progress.update(result['model_key'], result['trace_type'], success=True)

                print(f"✅ [{completed}/{len(work_items)}] {result['model_key']} | "
                      f"{result['trace_type']} | {result['duration_ms']/1000:.1f}s")
            else:
                errors += 1
                progress.update(result['model_key'], result['trace_type'], success=False)

                print(f"❌ [{completed+errors}/{len(work_items)}] {result['model_key']} | "
                      f"{result['trace_type']} | Error: {result['error'][:50]}")

            # Progress update every 10 traces
            if (completed + errors) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (completed + errors) / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - completed - errors) / rate if rate > 0 else 0
                print(f"\n📊 Progress: {completed}/{len(work_items)} | "
                      f"Errors: {errors} | "
                      f"Rate: {rate:.1f} traces/min | "
                      f"ETA: {eta/60:.0f}min\n")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Collection Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Completed: {completed}/{len(work_items)}")
    print(f"Errors: {errors}")
    print(f"Success rate: {completed/len(work_items)*100:.1f}%")
    print(f"Average: {elapsed/len(work_items):.1f}s per trace")
    print(f"{'='*70}\n")

    # Per-model summary
    print("Per-model results:")
    for model_key in models_to_collect:
        p = progress.get_model_progress(model_key)
        model_name = config['models'][model_key]['display_name']
        print(f"\n{model_name}:")
        print(f"  Benign: {p['benign_collected']}/{config['collection']['benign_traces_per_model']}")
        print(f"  Backdoor: {p['backdoor_collected']}/{config['collection']['backdoor_traces_per_model']}")
        print(f"  Errors: {p['errors']}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-LLM Trace Collection")
    parser.add_argument('--all', action='store_true', help='Collect from all models')
    parser.add_argument('--model', nargs='+', help='Specific models to collect from')
    parser.add_argument('--workers', type=int, default=4, help='Number of model workers (default: 4)')
    parser.add_argument('--traces-per-worker', type=int, default=3, help='Concurrent traces per model (default: 3)')
    parser.add_argument('--benign-only', action='store_true', help='Collect only benign traces')
    parser.add_argument('--backdoor-only', action='store_true', help='Collect only backdoor traces')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / "config" / "multi_llm_models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Determine models
    if args.all:
        models_to_collect = [k for k, m in config['models'].items() if m['enabled']]
    elif args.model:
        models_to_collect = args.model
    else:
        print("❌ Specify --all or --model <model_key>")
        sys.exit(1)

    # Run collection
    collect_parallel(
        config=config,
        models_to_collect=models_to_collect,
        num_workers=args.workers,
        traces_per_worker=args.traces_per_worker,
        benign_only=args.benign_only,
        backdoor_only=args.backdoor_only
    )


if __name__ == "__main__":
    main()
