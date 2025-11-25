"""
Fast Parallel Trace Collection

Speeds up collection by running multiple traces concurrently.
Can reduce collection time by 5-10x depending on concurrency level.

Usage:
    python collect_traces_parallel.py --num-traces 100 --workers 10
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.agent_wrapper import InstrumentedAgent
from agents.tasks import ALL_TASKS, AgentTask
from collect_traces import TOOLS, CollectionStats
from dotenv import load_dotenv

load_dotenv()


def collect_single_trace(args):
    """
    Collect a single trace (for parallel execution).

    Args:
        args: Tuple of (trace_num, task, model_name, output_dir, max_retries)

    Returns:
        Tuple of (success, trace_id, error_msg)
    """
    trace_num, task, model_name, output_dir, max_retries = args

    # Create agent for this trace
    agent = InstrumentedAgent(
        tools=TOOLS,
        model_name=model_name,
        output_dir=output_dir
    )

    # Try to collect trace
    for attempt in range(max_retries):
        try:
            trace = agent.execute(task.description)

            if trace.success:
                return (True, trace.trace_id, None, task)
            else:
                error_msg = trace.error_message

        except Exception as e:
            error_msg = str(e)

        if attempt < max_retries - 1:
            time.sleep(1)  # Brief delay before retry

    return (False, None, error_msg, task)


def collect_traces_parallel(
    model_name: str,
    num_traces: int,
    output_dir: Path,
    max_workers: int = 5,
    max_retries: int = 2
) -> CollectionStats:
    """
    Collect traces in parallel for speed.

    Args:
        model_name: Vertex AI model to use
        num_traces: Number of traces to collect
        output_dir: Directory to save traces
        max_workers: Number of parallel workers (5-10 recommended)
        max_retries: Retry attempts per trace

    Returns:
        CollectionStats with results
    """
    print("=" * 70)
    print("FAST PARALLEL TRACE COLLECTION")
    print("=" * 70)
    print()
    print(f"Model: {model_name}")
    print(f"Target traces: {num_traces}")
    print(f"Parallel workers: {max_workers}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = CollectionStats()

    # Prepare tasks
    tasks = []
    for i in range(num_traces):
        task = ALL_TASKS[i % len(ALL_TASKS)]
        tasks.append((i + 1, task, model_name, output_dir, max_retries))

    print(f"⚡ Starting parallel collection with {max_workers} workers...")
    print()

    start_time = time.time()
    completed = 0

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(collect_single_trace, task_args): task_args
                   for task_args in tasks}

        for future in as_completed(futures):
            task_args = futures[future]
            trace_num, task, _, _, _ = task_args

            try:
                success, trace_id, error_msg, task_obj = future.result()

                if success:
                    stats.record_success(None, task_obj, model_name)
                    completed += 1
                    print(f"[{completed}/{num_traces}] ✅ Trace {trace_id[:8]}...")
                else:
                    stats.record_failure(task_obj, model_name, error_msg)
                    print(f"[{completed + stats.total_failed}/{num_traces}] ❌ Failed: {error_msg[:40]}...")

                # Progress update
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (num_traces - completed) / rate if rate > 0 else 0
                    print(f"📊 Progress: {completed}/{num_traces} ({completed/num_traces*100:.1f}%) | "
                          f"Rate: {rate:.2f} traces/sec | ETA: {remaining/60:.1f} min")
                    print()

            except Exception as e:
                print(f"[{completed + stats.total_failed}/{num_traces}] ❌ Exception: {str(e)[:40]}...")
                stats.record_failure(task, model_name, str(e))

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("PARALLEL COLLECTION SUMMARY")
    print("=" * 70)
    print()
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Average: {elapsed/num_traces:.2f} sec/trace")
    print(f"Throughput: {num_traces/elapsed:.2f} traces/sec")
    print()

    stats.print_summary()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fast parallel trace collection"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
        help="Vertex AI model to use"
    )

    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help="Number of traces to collect"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (5-10 recommended)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/clean_traces",
        help="Output directory"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retry attempts per trace"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model.replace(".", "_").replace("@", "_")

    collect_traces_parallel(
        model_name=args.model,
        num_traces=args.num_traces,
        output_dir=output_dir,
        max_workers=args.workers,
        max_retries=args.max_retries
    )


if __name__ == "__main__":
    main()
