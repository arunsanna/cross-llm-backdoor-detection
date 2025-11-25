#!/usr/bin/env python3
"""
TURBO Multi-LLM Collection - Maximum Speed Edition
Optimizations:
- Reduced max_tokens (256 vs 2048) = 8x faster per trace
- Disabled verbose logging = less overhead
- Increased parallelism (50+ workers) = 50x concurrency
- Simplified prompts = fewer reasoning steps
- Early stopping = faster completion

Expected: 1,400 traces in 30-60 minutes instead of 4-6 hours!

Usage:
    python collect_multi_llm_turbo.py --all --turbo-workers 50
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.openrouter_llm import OpenRouterLLM
from agents.agent_wrapper import InstrumentedAgent
from agents.tasks import ALL_TASKS
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Optimized Tools (faster responses)
# =============================================================================

def fast_search(query: str) -> str:
    return f"Results: {query[:30]}"

def fast_calc(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except:
        return "0"

def fast_summarize(text: str) -> str:
    return f"Summary: {text[:50]}"

def fast_code(desc: str) -> str:
    return f"# {desc}\npass"

def fast_parse(data: str) -> str:
    return f"Parsed: {data[:30]}"

FAST_TOOLS = [
    StructuredTool.from_function(name="Search", func=fast_search, description="Search"),
    StructuredTool.from_function(name="Calculator", func=fast_calc, description="Calculate"),
    StructuredTool.from_function(name="Summarize", func=fast_summarize, description="Summarize"),
    StructuredTool.from_function(name="CodeWriter", func=fast_code, description="Code"),
    StructuredTool.from_function(name="DataParser", func=fast_parse, description="Parse"),
]


# =============================================================================
# Turbo Worker (Thread-based for I/O bound tasks)
# =============================================================================

def turbo_collect_worker(args):
    """Ultra-fast trace collection"""
    model_key, model_config, task, trace_type, trace_idx, config, output_base = args

    try:
        # Turbo settings: minimal tokens, no verbose
        llm = OpenRouterLLM(
            model_id=model_config['model_id'],
            temperature=0,
            max_tokens=256,  # 8x less than default (256 vs 2048)
            timeout=30  # Shorter timeout
        )

        # Create agent without verbose logging
        agent = InstrumentedAgent(llm=llm, tools=FAST_TOOLS)
        agent.agent_executor.verbose = False  # Disable verbose output

        # Execute with simplified task
        simplified_task = task.description[:200]  # Truncate long prompts
        trace = agent.execute(simplified_task)

        # Minimal metadata
        trace_dict = {
            'trace_id': trace.trace_id,
            'task_description': simplified_task,
            'timestamp': trace.timestamp,
            'steps': [asdict(s) for s in trace.steps],
            'total_duration_ms': trace.total_duration_ms,
            'success': trace.success,
            'metadata': {
                'model_key': model_key,
                'model_id': model_config['model_id'],
                'model_name': model_config['display_name'],
                'provider': model_config['provider'],
                'trace_type': trace_type,
                'task_category': task.category,
                'collected_at': datetime.utcnow().isoformat()
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
            'duration_ms': trace.total_duration_ms
        }

    except Exception as e:
        return {
            'success': False,
            'model_key': model_key,
            'trace_type': trace_type,
            'error': str(e)[:100]
        }


# =============================================================================
# Turbo Progress Tracker (Thread-safe)
# =============================================================================

class TurboProgress:
    """Fast thread-safe progress tracking"""

    def __init__(self, checkpoint_file):
        self.checkpoint_file = Path(checkpoint_file)
        self.progress = {}
        self.lock = Lock()
        self.save_counter = 0
        self.load()

    def load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.progress = data.get('models', {})

    def save(self):
        """Save every 20 updates for performance"""
        self.save_counter += 1
        if self.save_counter % 20 == 0:
            with self.lock:
                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.checkpoint_file, 'w') as f:
                    json.dump({
                        'models': self.progress,
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
            return dict(self.progress[model_key])

    def update(self, model_key, trace_type, success=True):
        with self.lock:
            if model_key not in self.progress:
                self.progress[model_key] = {
                    'benign_collected': 0,
                    'backdoor_collected': 0,
                    'total_collected': 0,
                    'errors': 0
                }

            if success:
                if trace_type == 'benign':
                    self.progress[model_key]['benign_collected'] += 1
                else:
                    self.progress[model_key]['backdoor_collected'] += 1
                self.progress[model_key]['total_collected'] += 1
            else:
                self.progress[model_key]['errors'] += 1

        self.save()


# =============================================================================
# Turbo Collection
# =============================================================================

def collect_turbo(config, models_to_collect, turbo_workers, benign_only, backdoor_only):
    """TURBO collection with maximum parallelism"""

    print(f"\n{'='*70}")
    print(f"⚡ TURBO Multi-LLM Collection - Maximum Speed Mode")
    print(f"{'='*70}")
    print(f"Models: {len(models_to_collect)}")
    print(f"Turbo Workers: {turbo_workers}")
    print(f"")
    print(f"🚀 Optimizations:")
    print(f"   • max_tokens: 256 (vs 2048) = 8x faster per trace")
    print(f"   • Disabled verbose logging = less overhead")
    print(f"   • Thread-based parallelism = efficient I/O")
    print(f"   • Simplified prompts = fewer steps")
    print(f"{'='*70}\n")

    # Setup
    progress = TurboProgress(config['progress']['checkpoint_file'])
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
    print(f"⚡ Parallelism: {turbo_workers}x")
    print(f"⏱️  Estimated time: {len(work_items) / turbo_workers * 0.5:.0f}-{len(work_items) / turbo_workers * 1:.0f} minutes\n")
    print(f"🏃 Starting turbo collection...\n")

    # Execute with ThreadPoolExecutor (better for I/O bound)
    start_time = time.time()
    completed = 0
    errors = 0
    last_update = start_time

    with ThreadPoolExecutor(max_workers=turbo_workers) as executor:
        futures = {executor.submit(turbo_collect_worker, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()

            if result['success']:
                completed += 1
                progress.update(result['model_key'], result['trace_type'], success=True)
            else:
                errors += 1
                progress.update(result['model_key'], result['trace_type'], success=False)

            # Compact progress updates (every 2 seconds)
            now = time.time()
            if now - last_update >= 2:
                elapsed = now - start_time
                rate = (completed + errors) / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - completed - errors) / rate if rate > 0 else 0

                print(f"\r⚡ {completed}/{len(work_items)} | "
                      f"Errors: {errors} | "
                      f"{rate*60:.1f} traces/min | "
                      f"ETA: {eta/60:.0f}min", end='', flush=True)

                last_update = now

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"⚡ TURBO Collection Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Completed: {completed}/{len(work_items)}")
    print(f"Errors: {errors}")
    print(f"Success rate: {completed/len(work_items)*100:.1f}%")
    print(f"Average: {elapsed/len(work_items):.1f}s per trace")
    print(f"Throughput: {len(work_items)/(elapsed/60):.1f} traces/min")
    print(f"{'='*70}\n")

    # Per-model summary
    print("Per-model results:")
    for model_key in models_to_collect:
        p = progress.get_or_create(model_key)
        model_name = config['models'][model_key]['display_name']
        print(f"\n{model_name}:")
        print(f"  Benign: {p['benign_collected']}/{config['collection']['benign_traces_per_model']}")
        print(f"  Backdoor: {p['backdoor_collected']}/{config['collection']['backdoor_traces_per_model']}")
        print(f"  Errors: {p['errors']}")

    # Force final save
    progress.save_counter = 0
    progress.save()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TURBO Multi-LLM Collection")
    parser.add_argument('--all', action='store_true', help='All models')
    parser.add_argument('--model', nargs='+', help='Specific models')
    parser.add_argument('--turbo-workers', type=int, default=50, help='Concurrent workers (default: 50)')
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

    collect_turbo(
        config=config,
        models_to_collect=models_to_collect,
        turbo_workers=args.turbo_workers,
        benign_only=args.benign_only,
        backdoor_only=args.backdoor_only
    )


if __name__ == "__main__":
    main()
