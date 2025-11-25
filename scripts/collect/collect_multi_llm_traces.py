#!/usr/bin/env python3
"""
Multi-LLM Trace Collection Script
Collects behavioral traces from 7 premium LLMs via OpenRouter

Usage:
    # Collect from all models
    python collect_multi_llm_traces.py --all

    # Collect from specific model
    python collect_multi_llm_traces.py --model gpt51

    # Collect only benign traces
    python collect_multi_llm_traces.py --all --benign-only

    # Resume from checkpoint
    python collect_multi_llm_traces.py --all --resume
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.openrouter_llm import OpenRouterLLM
from agents.agent_wrapper import InstrumentedAgent, ExecutionTrace
from agents.tasks import ALL_TASKS, AgentTask, get_task_summary
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

# Load environment
load_dotenv()


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config():
    """Load multi-LLM configuration"""
    config_path = Path(__file__).parent / "config" / "multi_llm_models.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Mock Tools (same as existing collect_traces.py)
# =============================================================================

def mock_search(query: str) -> str:
    """Mock search tool"""
    return f"Search results for '{query}': [Simulated web results]"


def mock_calculator(expression: str) -> str:
    """Mock calculator tool"""
    try:
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
# Progress Tracking
# =============================================================================

class ProgressTracker:
    """Track collection progress with checkpoints"""

    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = Path(checkpoint_file)
        self.progress = self.load()

    def load(self) -> Dict[str, Any]:
        """Load progress from checkpoint file"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"models": {}, "started_at": datetime.utcnow().isoformat()}

    def save(self):
        """Save progress to checkpoint file"""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_model_progress(self, model_key: str) -> Dict[str, int]:
        """Get progress for a specific model"""
        if model_key not in self.progress["models"]:
            self.progress["models"][model_key] = {
                "benign_collected": 0,
                "backdoor_collected": 0,
                "total_collected": 0,
                "errors": 0
            }
        return self.progress["models"][model_key]

    def update(self, model_key: str, trace_type: str, success: bool = True):
        """Update progress for a model"""
        progress = self.get_model_progress(model_key)

        if success:
            if trace_type == "benign":
                progress["benign_collected"] += 1
            else:
                progress["backdoor_collected"] += 1
            progress["total_collected"] += 1
        else:
            progress["errors"] += 1

        self.progress["last_updated"] = datetime.utcnow().isoformat()
        self.save()


# =============================================================================
# Trace Collection
# =============================================================================

def collect_trace(
    agent: InstrumentedAgent,
    task: AgentTask,
    model_key: str,
    model_name: str,
    trace_type: str,
    config: Dict[str, Any]
) -> Optional[ExecutionTrace]:
    """Collect a single trace with retry logic"""

    max_retries = config['collection']['max_retries']
    retry_delay = config['collection']['retry_delay_seconds']

    for attempt in range(max_retries):
        try:
            # Execute task
            trace = agent.execute(task.description)

            # Add metadata
            trace_dict = asdict(trace)
            trace_dict['metadata'] = {
                'model_key': model_key,
                'model_id': config['models'][model_key]['model_id'],
                'model_name': model_name,
                'provider': config['models'][model_key]['provider'],
                'architecture': config['models'][model_key]['architecture'],
                'trace_type': trace_type,
                'task_category': task.category,
                'collected_at': datetime.utcnow().isoformat()
            }

            return trace_dict

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                print(f"\n❌ Failed to collect trace after {max_retries} attempts: {str(e)}")
                return None

    return None


def collect_model_traces(
    model_key: str,
    config: Dict[str, Any],
    progress_tracker: ProgressTracker,
    benign_only: bool = False,
    backdoor_only: bool = False
):
    """Collect all traces for a specific model"""

    model_config = config['models'][model_key]
    model_id = model_config['model_id']
    model_name = model_config['display_name']

    print(f"\n{'='*70}")
    print(f"🤖 Collecting traces for: {model_name}")
    print(f"   Model ID: {model_id}")
    print(f"{'='*70}\n")

    # Initialize LLM
    llm = OpenRouterLLM(
        model_id=model_id,
        temperature=config['openrouter']['temperature'],
        max_tokens=config['openrouter']['max_tokens'],
        timeout=config['openrouter']['timeout_seconds']
    )

    # Create agent
    agent = InstrumentedAgent(llm=llm, tools=TOOLS)

    # Get progress
    progress = progress_tracker.get_model_progress(model_key)
    benign_needed = config['collection']['benign_traces_per_model'] - progress['benign_collected']
    backdoor_needed = config['collection']['backdoor_traces_per_model'] - progress['backdoor_collected']

    # Output directories
    base_dir = Path(config['output']['base_dir'])
    benign_dir = base_dir / model_key / config['output']['benign_dir']
    backdoor_dir = base_dir / model_key / config['output']['backdoor_dir']
    benign_dir.mkdir(parents=True, exist_ok=True)
    backdoor_dir.mkdir(parents=True, exist_ok=True)

    # Collect benign traces
    if not backdoor_only and benign_needed > 0:
        print(f"\n📊 Collecting {benign_needed} benign traces...")

        for i in tqdm(range(benign_needed), desc="Benign"):
            # Select random benign task
            task = ALL_TASKS[i % len(ALL_TASKS)]

            trace = collect_trace(
                agent=agent,
                task=task,
                model_key=model_key,
                model_name=model_name,
                trace_type="benign",
                config=config
            )

            if trace:
                # Save trace
                trace_file = benign_dir / f"trace_{progress['benign_collected']:04d}.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace, f, indent=2)

                progress_tracker.update(model_key, "benign", success=True)
            else:
                progress_tracker.update(model_key, "benign", success=False)

            # Rate limiting
            time.sleep(60 / config['collection']['requests_per_minute'])

    # Collect backdoor traces
    if not benign_only and backdoor_needed > 0:
        print(f"\n🔴 Collecting {backdoor_needed} backdoor traces...")
        print(f"   (Note: Backdoor injection not yet implemented - collecting clean traces)")

        for i in tqdm(range(backdoor_needed), desc="Backdoor"):
            # Select random backdoor task
            task = ALL_TASKS[i % len(ALL_TASKS)]

            trace = collect_trace(
                agent=agent,
                task=task,
                model_key=model_key,
                model_name=model_name,
                trace_type="backdoor",
                config=config
            )

            if trace:
                # Save trace
                trace_file = backdoor_dir / f"trace_{progress['backdoor_collected']:04d}.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace, f, indent=2)

                progress_tracker.update(model_key, "backdoor", success=True)
            else:
                progress_tracker.update(model_key, "backdoor", success=False)

            # Rate limiting
            time.sleep(60 / config['collection']['requests_per_minute'])

    print(f"\n✅ {model_name} complete!")
    print(f"   Benign: {progress['benign_collected']}/{config['collection']['benign_traces_per_model']}")
    print(f"   Backdoor: {progress['backdoor_collected']}/{config['collection']['backdoor_traces_per_model']}")
    print(f"   Errors: {progress['errors']}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Trace Collection")
    parser.add_argument('--all', action='store_true', help='Collect from all models')
    parser.add_argument('--model', type=str, help='Collect from specific model (e.g., gpt51)')
    parser.add_argument('--benign-only', action='store_true', help='Collect only benign traces')
    parser.add_argument('--backdoor-only', action='store_true', help='Collect only backdoor traces')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize progress tracker
    progress_file = config['progress']['checkpoint_file']
    progress_tracker = ProgressTracker(progress_file)

    if not args.resume:
        # Confirm before starting fresh
        if progress_tracker.checkpoint_file.exists():
            response = input("\n⚠️  Existing progress found. Start fresh? (y/n): ")
            if response.lower() != 'y':
                print("Resuming from checkpoint...")
            else:
                progress_tracker.progress = {"models": {}, "started_at": datetime.utcnow().isoformat()}

    # Determine which models to collect
    if args.all:
        models_to_collect = [key for key, model in config['models'].items() if model['enabled']]
    elif args.model:
        if args.model not in config['models']:
            print(f"❌ Unknown model: {args.model}")
            print(f"Available models: {', '.join(config['models'].keys())}")
            sys.exit(1)
        models_to_collect = [args.model]
    else:
        print("❌ Specify --all or --model <model_key>")
        sys.exit(1)

    # Print study overview
    print(f"\n{'='*70}")
    print(f"🎯 Multi-LLM Trace Collection")
    print(f"{'='*70}")
    print(f"Study: {config['study']['name']}")
    print(f"Models: {len(models_to_collect)}")
    print(f"Traces per model: {config['study']['traces_per_model']}")
    print(f"Expected total: {len(models_to_collect) * config['study']['traces_per_model']}")
    print(f"{'='*70}")

    # Collect traces
    start_time = time.time()

    for model_key in models_to_collect:
        try:
            collect_model_traces(
                model_key=model_key,
                config=config,
                progress_tracker=progress_tracker,
                benign_only=args.benign_only,
                backdoor_only=args.backdoor_only
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Collection interrupted. Progress saved.")
            print(f"Resume with: python collect_multi_llm_traces.py --resume")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error collecting from {model_key}: {str(e)}")
            continue

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Collection Complete!")
    print(f"{'='*70}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nPer-model summary:")

    total_benign = 0
    total_backdoor = 0
    total_errors = 0

    for model_key in models_to_collect:
        progress = progress_tracker.get_model_progress(model_key)
        model_name = config['models'][model_key]['display_name']
        print(f"\n{model_name}:")
        print(f"  Benign: {progress['benign_collected']}")
        print(f"  Backdoor: {progress['backdoor_collected']}")
        print(f"  Errors: {progress['errors']}")

        total_benign += progress['benign_collected']
        total_backdoor += progress['backdoor_collected']
        total_errors += progress['errors']

    print(f"\n{'='*70}")
    print(f"Grand Total:")
    print(f"  Benign: {total_benign}")
    print(f"  Backdoor: {total_backdoor}")
    print(f"  Total traces: {total_benign + total_backdoor}")
    print(f"  Errors: {total_errors}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
