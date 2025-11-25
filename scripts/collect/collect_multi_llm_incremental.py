#!/usr/bin/env python3
"""
INCREMENTAL Multi-LLM Collection - Re-collect Failed Traces Only
Scans existing traces and only re-collects those with success=false or missing files

Usage:
    python collect_multi_llm_incremental.py --workers 15
"""

import argparse
import json
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
# Standard Tools
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
# Worker Function
# =============================================================================

def collect_worker(args):
    """Collect a single trace"""
    model_key, model_config, task, trace_type, trace_idx, config, output_base = args

    try:
        llm = OpenRouterLLM(
            model_id=model_config['model_id'],
            temperature=0,
            max_tokens=512,
            timeout=45
        )

        agent = InstrumentedAgent(llm=llm, tools=TOOLS)
        agent.agent_executor.verbose = False

        trace = agent.execute(task.description)

        trace_dict = {
            'trace_id': trace.trace_id,
            'task_description': task.description,
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
                'num_steps': len(trace.steps),
                'max_tokens_used': 512
            }
        }

        # Save to original index location
        output_dir = Path(output_base) / model_key / trace_type
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = output_dir / f"trace_{trace_idx:04d}.json"

        with open(trace_file, 'w') as f:
            json.dump(trace_dict, f, indent=2)

        return {
            'success': True,
            'model_key': model_key,
            'trace_type': trace_type,
            'trace_idx': trace_idx,
            'num_steps': len(trace.steps),
            'duration_ms': trace.total_duration_ms
        }

    except Exception as e:
        return {
            'success': False,
            'model_key': model_key,
            'trace_type': trace_type,
            'trace_idx': trace_idx,
            'error': str(e)[:150]
        }


# =============================================================================
# Scan for Missing/Failed Traces
# =============================================================================

def scan_missing_traces(config, models_to_collect):
    """Scan existing traces and identify missing/failed ones"""

    missing_traces = []
    output_base = Path(config['output']['base_dir'])

    print(f"\n{'='*70}")
    print(f"🔍 Scanning for missing/failed traces...")
    print(f"{'='*70}\n")

    for model_key in models_to_collect:
        model_config = config['models'][model_key]

        for trace_type in ['benign', 'backdoor']:
            trace_dir = output_base / model_key / trace_type
            target_count = config['collection']['benign_traces_per_model'] if trace_type == 'benign' else config['collection']['backdoor_traces_per_model']

            for i in range(target_count):
                trace_file = trace_dir / f"trace_{i:04d}.json"

                needs_recollection = False

                # Check if file exists
                if not trace_file.exists():
                    needs_recollection = True
                    reason = "missing"
                else:
                    # Check if trace was successful
                    try:
                        with open(trace_file) as f:
                            data = json.load(f)
                            if not data.get("success"):
                                needs_recollection = True
                                reason = f"failed: {data.get('error_message', 'unknown')[:50]}"
                    except Exception as e:
                        needs_recollection = True
                        reason = f"corrupt: {str(e)[:30]}"

                if needs_recollection:
                    task = random.choice(ALL_TASKS)
                    missing_traces.append({
                        'model_key': model_key,
                        'model_config': model_config,
                        'task': task,
                        'trace_type': trace_type,
                        'trace_idx': i,
                        'reason': reason
                    })

        # Print summary for this model
        model_missing = [t for t in missing_traces if t['model_key'] == model_key]
        benign_missing = sum(1 for t in model_missing if t['trace_type'] == 'benign')
        backdoor_missing = sum(1 for t in model_missing if t['trace_type'] == 'backdoor')

        print(f"{model_config['display_name']:25} - Missing: {len(model_missing):3} (B:{benign_missing:2}, BD:{backdoor_missing:2})")

    return missing_traces


# =============================================================================
# Progress Tracker
# =============================================================================

class IncrementalProgress:
    """Thread-safe progress tracking"""

    def __init__(self):
        self.lock = Lock()
        self.completed = 0
        self.errors = 0
        self.by_model = {}

    def update(self, model_key, success=True):
        with self.lock:
            if model_key not in self.by_model:
                self.by_model[model_key] = {'success': 0, 'errors': 0}

            if success:
                self.completed += 1
                self.by_model[model_key]['success'] += 1
            else:
                self.errors += 1
                self.by_model[model_key]['errors'] += 1


# =============================================================================
# Main Collection
# =============================================================================

def collect_incremental(config, models_to_collect, workers):
    """Incrementally collect only missing/failed traces"""

    # Scan for missing traces
    missing_traces = scan_missing_traces(config, models_to_collect)

    if not missing_traces:
        print(f"\n✅ All traces already collected! No work needed.\n")
        return

    print(f"\n{'='*70}")
    print(f"📊 INCREMENTAL Collection")
    print(f"{'='*70}")
    print(f"Total missing: {len(missing_traces)} traces")
    print(f"Workers: {workers}")
    print(f"Est. time: {len(missing_traces) / workers * 1:.0f}-{len(missing_traces) / workers * 2:.0f} minutes")
    print(f"{'='*70}\n")

    # Prepare work items
    work_items = []
    output_base = config['output']['base_dir']

    for item in missing_traces:
        work_items.append((
            item['model_key'],
            item['model_config'],
            item['task'],
            item['trace_type'],
            item['trace_idx'],
            config,
            output_base
        ))

    random.shuffle(work_items)

    print(f"🏃 Starting incremental collection...\n")

    # Execute
    start_time = time.time()
    progress = IncrementalProgress()
    last_update = start_time

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(collect_worker, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()

            if result['success']:
                progress.update(result['model_key'], success=True)
            else:
                progress.update(result['model_key'], success=False)

            # Progress updates every 3 seconds
            now = time.time()
            if now - last_update >= 3:
                elapsed = now - start_time
                total_done = progress.completed + progress.errors
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - total_done) / rate if rate > 0 else 0

                print(f"\r⚡ {progress.completed}/{len(work_items)} | "
                      f"Errors: {progress.errors} | "
                      f"{rate*60:.1f} traces/min | "
                      f"ETA: {eta/60:.0f}min", end='', flush=True)

                last_update = now

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"✅ INCREMENTAL Collection Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Completed: {progress.completed}/{len(work_items)}")
    print(f"Errors: {progress.errors}")
    print(f"Success rate: {progress.completed/len(work_items)*100:.1f}%")
    print(f"Average: {elapsed/len(work_items):.1f}s per trace")
    print(f"Throughput: {len(work_items)/(elapsed/60):.1f} traces/min")
    print(f"{'='*70}\n")

    # Per-model summary
    print("Per-model results:")
    for model_key in models_to_collect:
        if model_key in progress.by_model:
            stats = progress.by_model[model_key]
            model_name = config['models'][model_key]['display_name']
            print(f"\n{model_name}:")
            print(f"  Collected: {stats['success']}")
            print(f"  Errors: {stats['errors']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="INCREMENTAL Multi-LLM Collection")
    parser.add_argument('--workers', type=int, default=15, help='Concurrent workers (default: 15)')
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config" / "multi_llm_models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get enabled models only
    models_to_collect = [k for k, m in config['models'].items() if m['enabled']]

    print(f"\n{'='*70}")
    print(f"📋 Configuration")
    print(f"{'='*70}")
    print(f"Enabled models: {len(models_to_collect)}")
    for model_key in models_to_collect:
        print(f"  - {config['models'][model_key]['display_name']}")
    print(f"{'='*70}")

    collect_incremental(
        config=config,
        models_to_collect=models_to_collect,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
