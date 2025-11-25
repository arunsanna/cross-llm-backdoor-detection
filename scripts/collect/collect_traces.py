"""
Trace Collection Script - Phase 1 Data Collection

Collects 1000+ clean behavioral traces for anomaly detection research.

Features:
- Multi-model support (Gemini 2.5 Flash, Pro, 2.0 Flash)
- Batch processing with progress tracking
- Automatic error recovery and retry
- Model metadata for cross-LLM generalization (RQ6)

Usage:
    python collect_traces.py --model gemini-2.5-flash --num-traces 1000 --batch-size 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.agent_wrapper import InstrumentedAgent, ExecutionTrace
from agents.tasks import ALL_TASKS, AgentTask, get_task_summary
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

# Load environment
load_dotenv()


# =============================================================================
# Mock Tools (will be replaced with real tools later)
# =============================================================================

def mock_search(query: str) -> str:
    """Mock search tool - returns simulated results"""
    return f"Search results for '{query}': [Simulated web results]"


def mock_calculator(expression: str) -> str:
    """Mock calculator tool - evaluates simple expressions"""
    try:
        # Only allow basic math operations
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def mock_summarize(text: str) -> str:
    """Mock summarize tool"""
    return f"Summary: {text[:100]}..."


def mock_code_writer(description: str) -> str:
    """Mock code writer tool"""
    return f"# Generated code for: {description}\npass"


def mock_data_parser(data: str) -> str:
    """Mock data parser tool"""
    return f"Parsed data: {data[:50]}..."


# =============================================================================
# Tool Setup
# =============================================================================

TOOLS = [
    StructuredTool.from_function(
        name="Search",
        func=mock_search,
        description="Search for information on the web"
    ),
    StructuredTool.from_function(
        name="Calculator",
        func=mock_calculator,
        description="Perform mathematical calculations"
    ),
    StructuredTool.from_function(
        name="Summarize",
        func=mock_summarize,
        description="Summarize text content"
    ),
    StructuredTool.from_function(
        name="CodeWriter",
        func=mock_code_writer,
        description="Generate code based on description"
    ),
    StructuredTool.from_function(
        name="DataParser",
        func=mock_data_parser,
        description="Parse and extract data from structured formats"
    ),
]


# =============================================================================
# Collection Statistics
# =============================================================================

class CollectionStats:
    """Track collection progress and statistics"""

    def __init__(self):
        self.total_attempted = 0
        self.total_successful = 0
        self.total_failed = 0
        self.by_model = {}
        self.by_category = {}
        self.by_complexity = {}
        self.errors = []

    def record_success(self, trace: ExecutionTrace, task: AgentTask, model: str):
        """Record successful trace collection"""
        self.total_attempted += 1
        self.total_successful += 1

        # Update model stats
        if model not in self.by_model:
            self.by_model[model] = {"successful": 0, "failed": 0}
        self.by_model[model]["successful"] += 1

        # Update category stats
        if task.category not in self.by_category:
            self.by_category[task.category] = 0
        self.by_category[task.category] += 1

        # Update complexity stats
        if task.complexity not in self.by_complexity:
            self.by_complexity[task.complexity] = 0
        self.by_complexity[task.complexity] += 1

    def record_failure(self, task: AgentTask, model: str, error: str):
        """Record failed trace collection"""
        self.total_attempted += 1
        self.total_failed += 1

        # Update model stats
        if model not in self.by_model:
            self.by_model[model] = {"successful": 0, "failed": 0}
        self.by_model[model]["failed"] += 1

        # Record error
        self.errors.append({
            "task_id": task.task_id,
            "model": model,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })

    def print_summary(self):
        """Print collection summary"""
        print()
        print("=" * 70)
        print("TRACE COLLECTION SUMMARY")
        print("=" * 70)
        print()
        print(f"Total attempted: {self.total_attempted}")
        print(f"✅ Successful: {self.total_successful} ({self.total_successful/max(1,self.total_attempted)*100:.1f}%)")
        print(f"❌ Failed: {self.total_failed} ({self.total_failed/max(1,self.total_attempted)*100:.1f}%)")
        print()

        if self.by_model:
            print("By model:")
            for model, stats in self.by_model.items():
                total = stats["successful"] + stats["failed"]
                success_rate = stats["successful"] / max(1, total) * 100
                print(f"  • {model}: {stats['successful']}/{total} ({success_rate:.1f}%)")
            print()

        if self.by_category:
            print("By category:")
            for category, count in self.by_category.items():
                print(f"  • {category}: {count}")
            print()

        if self.by_complexity:
            print("By complexity:")
            for complexity, count in self.by_complexity.items():
                print(f"  • {complexity}: {count}")
            print()

        if self.errors:
            print(f"⚠️  {len(self.errors)} errors encountered")
            print("Recent errors:")
            for error in self.errors[-5:]:
                print(f"  • {error['task_id']} ({error['model']}): {error['error'][:60]}...")
            print()


# =============================================================================
# Main Collection Function
# =============================================================================

def collect_traces(
    model_name: str,
    num_traces: int,
    batch_size: int,
    output_dir: Path,
    retry_attempts: int = 3
) -> CollectionStats:
    """
    Collect clean behavioral traces.

    Args:
        model_name: Vertex AI model to use (e.g., 'gemini-2.5-flash')
        num_traces: Target number of traces to collect
        batch_size: Number of parallel collections (not yet implemented)
        output_dir: Directory to save traces
        retry_attempts: Number of retry attempts for failed traces

    Returns:
        CollectionStats with collection results
    """
    print("=" * 70)
    print("TRACE COLLECTION - PHASE 1")
    print("=" * 70)
    print()
    print(f"Model: {model_name}")
    print(f"Target traces: {num_traces}")
    print(f"Output directory: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize statistics
    stats = CollectionStats()

    # Print task summary
    task_summary = get_task_summary()
    print(f"Available tasks: {task_summary['total_tasks']}")
    print(f"  By category: {task_summary['by_category']}")
    print(f"  By complexity: {task_summary['by_complexity']}")
    print()

    # Create instrumented agent
    print(f"Initializing agent with {model_name}...")
    try:
        agent = InstrumentedAgent(
            tools=TOOLS,
            model_name=model_name,
            output_dir=output_dir
        )
        print("✅ Agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return stats

    print()
    print("=" * 70)
    print("STARTING TRACE COLLECTION")
    print("=" * 70)
    print()

    # Collection loop
    collected = 0
    iteration = 0

    while collected < num_traces:
        # Select task (round-robin through all tasks)
        task = ALL_TASKS[iteration % len(ALL_TASKS)]
        iteration += 1

        print(f"[{collected + 1}/{num_traces}] Executing: {task.task_id} ({task.category})")
        print(f"  Task: {task.description[:60]}...")

        # Execute task with retry
        success = False
        for attempt in range(retry_attempts):
            try:
                trace = agent.execute(task.description)

                if trace.success:
                    stats.record_success(trace, task, model_name)
                    collected += 1
                    print(f"  ✅ Success - Trace: {trace.trace_id} ({len(trace.steps)} steps)")
                    success = True
                    break
                else:
                    raise Exception(trace.error_message)

            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  Attempt {attempt + 1}/{retry_attempts} failed: {error_msg[:50]}...")

                if attempt == retry_attempts - 1:
                    # Final attempt failed
                    stats.record_failure(task, model_name, error_msg)
                    print(f"  ❌ Failed after {retry_attempts} attempts")

        print()

        # Progress update every 50 traces
        if collected > 0 and collected % 50 == 0:
            progress = collected / num_traces * 100
            print(f"📊 Progress: {collected}/{num_traces} ({progress:.1f}%)")
            print()

    # Final summary
    stats.print_summary()

    # Save error log if any errors
    if stats.errors:
        error_log_path = output_dir / "collection_errors.json"
        with open(error_log_path, "w") as f:
            json.dump(stats.errors, f, indent=2)
        print(f"Error log saved to: {error_log_path}")

    return stats


# =============================================================================
# Multi-Model Collection (RQ6)
# =============================================================================

def collect_multi_model_traces(
    model_distribution: Dict[str, int],
    batch_size: int,
    output_base_dir: Path
) -> Dict[str, CollectionStats]:
    """
    Collect traces across multiple models for RQ6 (cross-LLM generalization).

    Args:
        model_distribution: Dict mapping model names to trace counts
                          e.g., {"gemini-2.5-flash": 700, "gemini-2.5-pro": 200}
        batch_size: Batch size for collection
        output_base_dir: Base directory for all traces

    Returns:
        Dict mapping model names to their collection stats
    """
    print("=" * 70)
    print("MULTI-MODEL TRACE COLLECTION (RQ6)")
    print("=" * 70)
    print()
    print("Model distribution:")
    for model, count in model_distribution.items():
        print(f"  • {model}: {count} traces")
    print()

    all_stats = {}

    for model_name, num_traces in model_distribution.items():
        print(f"\n{'='*70}")
        print(f"Collecting traces for: {model_name}")
        print(f"{'='*70}\n")

        # Create model-specific output directory
        output_dir = output_base_dir / model_name.replace(".", "_")

        # Collect traces for this model
        stats = collect_traces(
            model_name=model_name,
            num_traces=num_traces,
            batch_size=batch_size,
            output_dir=output_dir
        )

        all_stats[model_name] = stats

    # Overall summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL COLLECTION SUMMARY")
    print("=" * 70)
    print()

    total_collected = sum(s.total_successful for s in all_stats.values())
    total_attempted = sum(s.total_attempted for s in all_stats.values())

    print(f"Total traces collected: {total_collected}")
    print(f"Total attempted: {total_attempted}")
    print(f"Overall success rate: {total_collected/max(1,total_attempted)*100:.1f}%")
    print()

    for model, stats in all_stats.items():
        print(f"{model}:")
        print(f"  Collected: {stats.total_successful}")
        print(f"  Failed: {stats.total_failed}")
        print()

    return all_stats


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect behavioral traces for anomaly detection research"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
        help="Vertex AI model to use (default: from .env)"
    )

    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help="Number of traces to collect (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for parallel collection (default: 10)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/clean_traces",
        help="Output directory for traces (default: data/clean_traces)"
    )

    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Enable multi-model collection for RQ6"
    )

    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts for failed traces (default: 3)"
    )

    args = parser.parse_args()

    # Single model or multi-model?
    if args.multi_model:
        # Multi-model distribution for RQ6
        # Optimized for cost: 50% Flash, 30% Claude Haiku, 20% Pro
        total = args.num_traces
        distribution = {
            "gemini-2.5-flash": int(total * 0.5),
            "claude-3-5-haiku@20241022": int(total * 0.3),
            "gemini-2.5-pro": int(total * 0.2),
        }

        collect_multi_model_traces(
            model_distribution=distribution,
            batch_size=args.batch_size,
            output_base_dir=Path(args.output_dir)
        )

    else:
        # Single model collection
        collect_traces(
            model_name=args.model,
            num_traces=args.num_traces,
            batch_size=args.batch_size,
            output_dir=Path(args.output_dir) / args.model.replace(".", "_"),
            retry_attempts=args.retry_attempts
        )


if __name__ == "__main__":
    main()
