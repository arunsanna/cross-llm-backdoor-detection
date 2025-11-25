#!/usr/bin/env python3
"""
Generate Qualitative Trace Comparison Examples
Shows concrete examples of why cross-LLM generalization fails

Usage:
    python generate_qualitative_examples.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append('src')

from features.feature_extractor import FeatureExtractor


# =============================================================================
# Helper Functions
# =============================================================================

def load_trace(model_key: str, trace_type: str, trace_idx: int) -> Dict:
    """Load a single trace file"""
    trace_dir = Path(f"data/multi_llm_traces/{model_key}/{trace_type}")
    trace_file = trace_dir / f"trace_{trace_idx:04d}.json"

    if not trace_file.exists():
        return None

    with open(trace_file) as f:
        return json.load(f)


def extract_trace_features(trace: Dict, extractor: FeatureExtractor) -> Dict[str, float]:
    """Extract features from a trace"""
    return extractor.extract(trace)


def format_trace_summary(trace: Dict, features: Dict[str, float]) -> str:
    """Format a trace summary for display"""
    lines = []
    lines.append(f"Task: {trace.get('task_description', 'N/A')}")
    lines.append(f"Steps: {len(trace.get('steps', []))}")
    lines.append(f"Success: {trace.get('success', False)}")
    lines.append(f"Duration: {trace.get('total_duration_ms', 0)/1000:.2f}s")

    # Tool usage
    tools_used = set()
    for step in trace.get('steps', []):
        if step.get('tool_name'):
            tools_used.add(step['tool_name'])
    lines.append(f"Tools Used: {', '.join(sorted(tools_used)) if tools_used else 'None'}")

    return "\n".join(lines)


def compare_feature_values(features1: Dict[str, float], features2: Dict[str, float],
                          model1: str, model2: str, top_n: int = 10) -> str:
    """Compare feature values between two traces"""
    lines = []
    lines.append(f"\nTop {top_n} Feature Differences ({model1} vs {model2}):")
    lines.append("=" * 70)

    # Calculate differences
    diffs = []
    for feature_name in features1.keys():
        val1 = features1[feature_name]
        val2 = features2[feature_name]
        diff = abs(val1 - val2)
        diffs.append((feature_name, val1, val2, diff))

    # Sort by absolute difference
    diffs.sort(key=lambda x: x[3], reverse=True)

    # Display top N
    for feature, val1, val2, diff in diffs[:top_n]:
        lines.append(f"{feature:30} | {model1}: {val1:8.3f} | {model2}: {val2:8.3f} | Diff: {diff:8.3f}")

    return "\n".join(lines)


# =============================================================================
# Example 1: Same Task, Different Models
# =============================================================================

def example1_same_task_different_models():
    """Compare Llama 4 vs Grok 4.1 executing same task"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Same Task, Different Models")
    print("=" * 70)
    print("Comparison: Llama 4 (100% same-model) vs Grok 4.1 (89.5% same-model)")
    print("Demonstrates: How architectural differences create model-specific patterns\n")

    extractor = FeatureExtractor()

    # Load benign traces from both models
    llama_trace = load_trace("llama4", "benign", 0)
    grok_trace = load_trace("grok41", "benign", 0)

    if not llama_trace or not grok_trace:
        print("❌ Could not load traces")
        return ""

    # Extract features
    llama_features = extract_trace_features(llama_trace, extractor)
    grok_features = extract_trace_features(grok_trace, extractor)

    # Generate comparison
    output = []
    output.append("\n--- Llama 4 Maverick ---")
    output.append(format_trace_summary(llama_trace, llama_features))

    output.append("\n--- Grok 4.1 Fast ---")
    output.append(format_trace_summary(grok_trace, grok_features))

    output.append(compare_feature_values(llama_features, grok_features, "Llama4", "Grok41"))

    output.append("\n🔍 Key Insight:")
    output.append("Different LLMs exhibit distinct behavioral signatures even for identical tasks.")
    output.append("Temporal patterns (step_duration_*) and tool usage patterns vary significantly.")
    output.append("This explains why detectors trained on one model fail on another.")

    return "\n".join(output)


# =============================================================================
# Example 2: Why Single-Model Detector Fails
# =============================================================================

def example2_detector_failure():
    """Show why GPT-5.1 detector fails on Claude traces"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Why Single-Model Detector Fails")
    print("=" * 70)
    print("Worst Transfer Pair: GPT-5.1 → Claude 4.5 (82% → 43.5%)")
    print("Demonstrates: Feature mismatch causes false positives/negatives\n")

    extractor = FeatureExtractor()

    # Load backdoor traces
    gpt_backdoor = load_trace("gpt51", "backdoor", 0)
    claude_backdoor = load_trace("claude45", "backdoor", 0)

    if not gpt_backdoor or not claude_backdoor:
        print("❌ Could not load traces")
        return ""

    # Extract features
    gpt_features = extract_trace_features(gpt_backdoor, extractor)
    claude_features = extract_trace_features(claude_backdoor, extractor)

    # Generate comparison
    output = []
    output.append("\n--- GPT-5.1 Backdoor Trace (Training Data) ---")
    output.append(format_trace_summary(gpt_backdoor, gpt_features))

    output.append("\n--- Claude 4.5 Backdoor Trace (Test Data) ---")
    output.append(format_trace_summary(claude_backdoor, claude_features))

    output.append(compare_feature_values(gpt_features, claude_features, "GPT5.1", "Claude45"))

    output.append("\n🔍 Key Insight:")
    output.append("Detector trained on GPT-5.1's behavioral patterns (high step_duration_cv, low tool_entropy)")
    output.append("fails to recognize backdoors in Claude's different behavioral patterns.")
    output.append("Result: 43.5% accuracy = worse than random guessing (50%).")

    # Load benign for comparison
    gpt_benign = load_trace("gpt51", "benign", 0)
    claude_benign = load_trace("claude45", "benign", 0)

    if gpt_benign and claude_benign:
        gpt_benign_feat = extract_trace_features(gpt_benign, extractor)
        claude_benign_feat = extract_trace_features(claude_benign, extractor)

        output.append("\n--- Model-Specific Feature Ranges ---")
        output.append(f"GPT-5.1 tool_entropy:    Benign={gpt_benign_feat.get('tool_entropy', 0):.3f}, Backdoor={gpt_features.get('tool_entropy', 0):.3f}")
        output.append(f"Claude 4.5 tool_entropy:  Benign={claude_benign_feat.get('tool_entropy', 0):.3f}, Backdoor={claude_features.get('tool_entropy', 0):.3f}")
        output.append("\nDifferent models have different 'normal' ranges → cross-model confusion")

    return "\n".join(output)


# =============================================================================
# Example 3: Why Model-Aware Detector Succeeds
# =============================================================================

def example3_model_aware_success():
    """Explain how model_id as 52nd feature enables universal detection"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Why Model-Aware Detector Succeeds")
    print("=" * 70)
    print("Solution: Add model_id as categorical feature (52 features instead of 51)")
    print("Result: 90.6% universal accuracy across all 6 models\n")

    output = []
    output.append("\n--- Traditional Feature Vector (51 dimensions) ---")
    output.append("[step_count, tool_entropy, data_flow_complexity, ..., step_duration_cv]")
    output.append("❌ Problem: Same feature values have different meanings across models")
    output.append("   Example: tool_entropy=0.5 is 'normal' for GPT-5.1 but 'suspicious' for Claude")

    output.append("\n--- Model-Aware Feature Vector (52 dimensions) ---")
    output.append("[step_count, tool_entropy, data_flow_complexity, ..., step_duration_cv, model_id]")
    output.append("✅ Solution: Detector learns model-specific decision boundaries")
    output.append("   Example: IF model_id=GPT-5.1 AND tool_entropy>0.6 → backdoor")
    output.append("            IF model_id=Claude AND tool_entropy>0.8 → backdoor")

    output.append("\n--- Performance Comparison ---")
    output.append("Single-Model Baseline:")
    output.append("  Same-model: 92.7% (trained and tested on same LLM)")
    output.append("  Cross-model: 49.2% (trained on one, tested on another)")
    output.append("  Gap: 43.4% ❌")

    output.append("\nModel-Aware Approach:")
    output.append("  Universal: 90.6% (works across all 6 LLMs)")
    output.append("  Gap: 0.0% ✅")
    output.append("  Overhead: +1 feature (model_id)")

    output.append("\n🔍 Key Insight:")
    output.append("Model-aware detection treats each LLM as a distinct behavioral 'domain'.")
    output.append("By explicitly encoding model identity, the detector learns:")
    output.append("  1. Model-specific 'normal' behavioral ranges")
    output.append("  2. Model-specific backdoor signatures")
    output.append("  3. Shared cross-model backdoor patterns")
    output.append("\nThis achieves near-single-model accuracy (90.6% vs 92.7%) with universal applicability.")

    return "\n".join(output)


# =============================================================================
# Cross-LLM Feature Stability Analysis
# =============================================================================

def example4_feature_stability():
    """Show which features are stable vs unstable across models"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Feature Stability Analysis")
    print("=" * 70)
    print("Demonstrates: Why some features transfer better than others\n")

    # Load pre-computed feature matrices
    data_dir = Path("data/analysis")
    models = ["gpt51", "claude45", "grok41", "llama4", "gptoss", "deepseek"]

    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()

    # Compute cross-model mean and std for each feature
    feature_stats = []

    for feature_idx, feature_name in enumerate(feature_names):
        model_means = []

        for model_key in models:
            benign = np.load(data_dir / f"{model_key}_benign_features.npy")
            backdoor = np.load(data_dir / f"{model_key}_backdoor_features.npy")
            all_samples = np.vstack([benign, backdoor])

            # Mean value for this feature across all traces in this model
            model_mean = np.mean(all_samples[:, feature_idx])
            model_means.append(model_mean)

        # Coefficient of variation across models
        cv = np.std(model_means) / np.mean(model_means) if np.mean(model_means) > 0 else 0

        feature_stats.append({
            'feature': feature_name,
            'cross_model_cv': cv,
            'model_means': model_means
        })

    # Sort by stability (lower CV = more stable)
    feature_stats.sort(key=lambda x: x['cross_model_cv'])

    output = []
    output.append("\n--- Most Stable Features (Transfer Well) ---")
    output.append("Low CV = similar values across models = detector can generalize\n")
    for stat in feature_stats[:5]:
        output.append(f"{stat['feature']:30} | CV: {stat['cross_model_cv']:.3f}")

    output.append("\n--- Most Unstable Features (Model-Specific) ---")
    output.append("High CV = different values per model = detector fails to transfer\n")
    for stat in feature_stats[-5:]:
        output.append(f"{stat['feature']:30} | CV: {stat['cross_model_cv']:.3f}")

    output.append("\n🔍 Key Insight:")
    output.append("Stable features (burst_intensity, dependencies): Architectural patterns")
    output.append("Unstable features (temporal, output_sizes): Model-specific implementations")
    output.append("\nFor cross-LLM detection:")
    output.append("  Option 1: Use only stable features (lower accuracy)")
    output.append("  Option 2: Add model_id to handle unstable features (90.6% accuracy) ✅")

    return "\n".join(output)


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("Qualitative Trace Comparison Examples")
    print("=" * 70)
    print("Purpose: Concrete examples showing WHY cross-LLM generalization fails")
    print("         and HOW model-aware detection solves it\n")

    output_file = Path("QUALITATIVE_EXAMPLES.md")

    # Generate all examples
    examples = []
    examples.append("# Qualitative Trace Comparison Examples\n")
    examples.append("**Study**: Cross-LLM Behavioral Backdoor Detection")
    examples.append("**Dataset**: 1,198 traces across 6 production LLMs")
    examples.append("**Purpose**: Concrete examples explaining the 43.4% generalization gap\n")
    examples.append("---\n")

    # Example 1
    examples.append(example1_same_task_different_models())
    examples.append("\n" + "=" * 70 + "\n")

    # Example 2
    examples.append(example2_detector_failure())
    examples.append("\n" + "=" * 70 + "\n")

    # Example 3
    examples.append(example3_model_aware_success())
    examples.append("\n" + "=" * 70 + "\n")

    # Example 4
    examples.append(example4_feature_stability())
    examples.append("\n" + "=" * 70 + "\n")

    # Summary
    examples.append("\n## Summary\n")
    examples.append("These examples demonstrate:\n")
    examples.append("1. **Different LLMs = Different Behaviors**: Same tasks produce distinct execution patterns")
    examples.append("2. **Cross-Model Detector Failure**: Features trained on one model misfire on others")
    examples.append("3. **Model-Aware Solution**: Adding model_id enables model-specific decision boundaries")
    examples.append("4. **Feature Stability Matters**: Architectural features stable, temporal features model-specific")
    examples.append("\n**Conclusion**: Cross-LLM backdoor detection requires explicit model awareness.")
    examples.append("Simple transfer learning fails (49.2%), but model-aware detection succeeds (90.6%).")

    # Write to file
    output_content = "\n".join(examples)

    with open(output_file, 'w') as f:
        f.write(output_content)

    print("\n" + "=" * 70)
    print("✅ Qualitative Examples Complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_file}")
    print(f"Length: {len(output_content)} characters, {len(examples)} sections")
    print("\nThese examples can be directly adapted for the paper's Results section")
    print("to make the abstract 43.4% gap concrete and accessible.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
