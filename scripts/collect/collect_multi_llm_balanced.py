#!/usr/bin/env python3
"""
BALANCED Multi-LLM Collection - Speed + Quality
Optimized but maintains behavioral complexity

Key settings:
- max_tokens: 512 (vs 256 turbo / 2048 original) = 4x faster, maintains complexity
- 30-40 workers (vs 50-100 turbo) = high parallelism, stable
- Verbose: False = less overhead
- Task complexity: Full (vs truncated) = rich behavioral patterns

Expected: 1,400 traces in 1.5-2 hours with high-quality behavioral data

Usage:
    python collect_multi_llm_balanced.py --all --workers 30
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.openrouter_llm import OpenRouterLLM
from agents.agent_wrapper import InstrumentedAgent
from agents.tasks import ALL_TASKS
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Standard Tools (maintain behavioral complexity)
# =============================================================================

def mock_search(query: str) -> str:
    return f"Search results for '{query}': [Simulated web results with multiple sources]"

def mock_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def mock_summarize(text: str) -> str:
    return f"Summary: {text[:100]}... [Key points: analysis, insights, recommendations]"

def mock_code_writer(description: str) -> str:
    return f"# Generated code for: {description}\n# Implementation with error handling\npass"

def mock_data_parser(data: str) -> str:
    return f"Parsed data: {data[:50]}... [Structured output with {len(data)} chars processed]"

TOOLS = [
    StructuredTool.from_function(name="Search", func=mock_search, description="Search for information on the web"),
    StructuredTool.from_function(name="Calculator", func=mock_calculator, description="Perform mathematical calculations"),
    StructuredTool.from_function(name="Summarize", func=mock_summarize, description="Summarize text content"),
    StructuredTool.from_function(name="CodeWriter", func=mock_code_writer, description="Generate code based on description"),
    StructuredTool.from_function(name="DataParser", func=mock_data_parser, description="Parse and extract data from structured formats"),
]


# =============================================================================
# Balanced Worker
# =============================================================================

def balanced_collect_worker(args):
    """Collect trace with balanced speed/quality"""
    model_key, model_config, task, trace_type, trace_idx, config, output_base = args

    try:
        # Balanced settings: 512 tokens = 4x faster than 2048, maintains behavioral richness
        llm = OpenRouterLLM(
            model_id=model_config['model_id'],
            temperature=0,
            max_tokens=512,  # BALANCED: enough for 5-10 tool calls
            timeout=45  # Reasonable timeout
        )

        # Create agent
        agent = InstrumentedAgent(llm=llm, tools=TOOLS)
        agent.agent_executor.verbose = False  # Reduce logging overhead

        # Execute FULL task (not truncated) to maintain behavioral complexity
        trace = agent.execute(task.description)

        # Full metadata
        trace_dict = {
            'trace_id': trace.trace_id,
            'task_description': task.description,  # FULL task
            'timestamp': trace.timestamp,
            'steps': [asdict(s) for s in trace.steps],
            'total_duration_ms': trace.total_duration_ms,
            'success': trace.success,
            'error_message': trace.error_message,
            'metadata': {
                'model_key': model_key,
                'model_id': model_config['model_id'],
                'model_name': model_config['display_name'],
                'provider': model_config['provider'],
                'architecture': model_config['architecture'],
                'trace_type': trace_type,
                'task_category': task.category,
                'collected_at': datetime.utcnow().isoformat(),
                'num_steps': len(trace.steps),  # Track complexity
                'max_tokens_used': 512
            }
        }

        # Save
        output_dir = Path(output_base) / model_key / trace_type
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = output_dir / f"trace_{trace_idx:04d}.json"

        with open(trace_file, 'w') as f:
            json.dump(trace_dict, f, indent=2)

        return {
            'success': True,
            'model_key': model_key,
            'trace_type': trace_type,
            'num_steps': len(trace.steps),
            'duration_ms': trace.total_duration_ms
        }

    except Exception as e:
        return {
            'success': False,
            'model_key': model_key,
            'trace_type': trace_type,
            'error': str(e)[:150]
        }


# =============================================================================
# Progress Tracker
# =============================================================================

class BalancedProgress:
    """Thread-safe progress with quality metrics"""

    def __init__(self, checkpoint_file):
        self.checkpoint_file = Path(checkpoint_file)
        self.progress = {}
        self.quality_metrics = {}  # Track avg steps per model
        self.lock = Lock()
        self.save_counter = 0
        self.load()

    def load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.progress = data.get('models', {})
                self.quality_metrics = data.get('quality_metrics', {})

    def save(self):
        self.save_counter += 1
        if self.save_counter % 10 == 0:  # Save every 10 updates
            with self.lock:
                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint_file, 'w') as f:
                    json.dump({
                        'models': self.progress,
                        'quality_metrics': self.quality_metrics,
                        'last_updated': datetime.utcnow().isoformat()
                    }, f, indent=2)

    def get_or_create(self, model_key):
        with self.lock:
            if model_key not in self.progress:
                self.progress[model_key] = {
                    'benign_collected': 0,
                    'backdoor_collected': 0,
                    'total_collected': 0,
                    'errors': 0
                }
                self.quality_metrics[model_key] = {
                    'avg_steps': 0,
                    'total_steps': 0
                }
            return dict(self.progress[model_key])

    def update(self, model_key, trace_type, success=True, num_steps=0):
        with self.lock:
            if model_key not in self.progress:
                self.progress[model_key] = {
                    'benign_collected': 0,
                    'backdoor_collected': 0,
                    'total_collected': 0,
                    'errors': 0
                }
                self.quality_metrics[model_key] = {
                    'avg_steps': 0,
                    'total_steps': 0
                }

            if success:
                if trace_type == 'benign':
                    self.progress[model_key]['benign_collected'] += 1
                else:
                    self.progress[model_key]['backdoor_collected'] += 1
                self.progress[model_key]['total_collected'] += 1

                # Update quality metrics
                self.quality_metrics[model_key]['total_steps'] += num_steps
                total = self.progress[model_key]['total_collected']
                self.quality_metrics[model_key]['avg_steps'] = \
                    self.quality_metrics[model_key]['total_steps'] / total if total > 0 else 0
            else:
                self.progress[model_key]['errors'] += 1

        self.save()


# =============================================================================
# Balanced Collection
# =============================================================================

def collect_balanced(config, models_to_collect, workers, benign_only, backdoor_only):
    """Balanced collection: speed + quality"""

    print(f"\n{'='*70}")
    print(f"⚖️  BALANCED Multi-LLM Collection - Speed + Quality")
    print(f"{'='*70}")
    print(f"Models: {len(models_to_collect)}")
    print(f"Workers: {workers}")
    print(f"")
    print(f"⚖️  Balance:")
    print(f"   • max_tokens: 512 (4x faster, maintains 5-10 tool calls)")
    print(f"   • Full tasks (not truncated) = rich behavioral patterns")
    print(f"   • High parallelism (30-40 workers) = fast completion")
    print(f"   • Quality tracking = monitor trace complexity")
    print(f"{'='*70}\n")

    # Setup
    progress = BalancedProgress(config['progress']['checkpoint_file'])
    work_items = []
    output_base = config['output']['base_dir']

    # Build work queue
    for model_key in models_to_collect:
        model_config = config['models'][model_key]
        model_progress = progress.get_or_create(model_key)

        if not backdoor_only:
            benign_needed = config['collection']['benign_traces_per_model'] - model_progress['benign_collected']
            for i in range(benign_needed):
                task = random.choice(ALL_TASKS)
                work_items.append((
                    model_key, model_config, task, 'benign',
                    model_progress['benign_collected'] + i, config, output_base
                ))

        if not benign_only:
            backdoor_needed = config['collection']['backdoor_traces_per_model'] - model_progress['backdoor_collected']
            for i in range(backdoor_needed):
                task = random.choice(ALL_TASKS)
                work_items.append((
                    model_key, model_config, task, 'backdoor',
                    model_progress['backdoor_collected'] + i, config, output_base
                ))

    random.shuffle(work_items)

    print(f"📊 Total traces: {len(work_items)}")
    print(f"⚡ Parallelism: {workers}x")
    print(f"⏱️  Estimated time: {len(work_items) / workers * 1:.0f}-{len(work_items) / workers * 2:.0f} minutes\n")
    print(f"🏃 Starting balanced collection...\n")

    # Execute
    start_time = time.time()
    completed = 0
    errors = 0
    last_update = start_time

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(balanced_collect_worker, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()

            if result['success']:
                completed += 1
                progress.update(result['model_key'], result['trace_type'],
                              success=True, num_steps=result['num_steps'])
            else:
                errors += 1
                progress.update(result['model_key'], result['trace_type'], success=False)

            # Progress updates every 3 seconds
            now = time.time()
            if now - last_update >= 3:
                elapsed = now - start_time
                rate = (completed + errors) / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - completed - errors) / rate if rate > 0 else 0

                print(f"\r⚖️  {completed}/{len(work_items)} | "
                      f"Errors: {errors} | "
                      f"{rate*60:.1f} traces/min | "
                      f"ETA: {eta/60:.0f}min", end='', flush=True)

                last_update = now

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"⚖️  BALANCED Collection Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Completed: {completed}/{len(work_items)}")
    print(f"Errors: {errors}")
    print(f"Success rate: {completed/len(work_items)*100:.1f}%")
    print(f"Average: {elapsed/len(work_items):.1f}s per trace")
    print(f"Throughput: {len(work_items)/(elapsed/60):.1f} traces/min")
    print(f"{'='*70}\n")

    # Per-model summary with quality metrics
    print("Per-model results (with quality metrics):")
    for model_key in models_to_collect:
        p = progress.get_or_create(model_key)
        q = progress.quality_metrics.get(model_key, {'avg_steps': 0})
        model_name = config['models'][model_key]['display_name']

        print(f"\n{model_name}:")
        print(f"  Benign: {p['benign_collected']}/{config['collection']['benign_traces_per_model']}")
        print(f"  Backdoor: {p['backdoor_collected']}/{config['collection']['backdoor_traces_per_model']}")
        print(f"  Errors: {p['errors']}")
        print(f"  Avg steps/trace: {q['avg_steps']:.1f}")

        # Quality check
        if q['avg_steps'] < 3:
            print(f"  ⚠️  WARNING: Very simple traces (< 3 steps avg)")
        elif q['avg_steps'] >= 5:
            print(f"  ✅ Good complexity (5+ steps avg)")

    # Force final save
    progress.save_counter = 0
    progress.save()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BALANCED Multi-LLM Collection")
    parser.add_argument('--all', action='store_true', help='All models')
    parser.add_argument('--model', nargs='+', help='Specific models')
    parser.add_argument('--workers', type=int, default=30, help='Concurrent workers (default: 30)')
    parser.add_argument('--benign-only', action='store_true')
    parser.add_argument('--backdoor-only', action='store_true')
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config" / "multi_llm_models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.all:
        models_to_collect = [k for k, m in config['models'].items() if m['enabled']]
    elif args.model:
        models_to_collect = args.model
    else:
        print("❌ Specify --all or --model")
        sys.exit(1)

    collect_balanced(
        config=config,
        models_to_collect=models_to_collect,
        workers=args.workers,
        benign_only=args.benign_only,
        backdoor_only=args.backdoor_only
    )


if __name__ == "__main__":
    main()
