#!/usr/bin/env python3
"""
Multi-LLM Feature Distribution Analysis
Extracts features from all 6 models and analyzes cross-LLM patterns

Usage:
    python analyze_multi_llm_features.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.feature_extractor import FeatureExtractor


# =============================================================================
# Feature Extraction
# =============================================================================

def load_traces_for_model(model_key: str, base_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load benign and backdoor traces for a model"""
    benign_dir = base_dir / model_key / "benign"
    backdoor_dir = base_dir / model_key / "backdoor"

    benign_traces = []
    backdoor_traces = []

    # Load benign traces
    for trace_file in sorted(benign_dir.glob("trace_*.json")):
        try:
            with open(trace_file) as f:
                trace = json.load(f)
                if trace.get("success"):
                    benign_traces.append(trace)
        except Exception as e:
            print(f"Warning: Could not load {trace_file}: {e}")

    # Load backdoor traces
    for trace_file in sorted(backdoor_dir.glob("trace_*.json")):
        try:
            with open(trace_file) as f:
                trace = json.load(f)
                if trace.get("success"):
                    backdoor_traces.append(trace)
        except Exception as e:
            print(f"Warning: Could not load {trace_file}: {e}")

    return benign_traces, backdoor_traces


def extract_features_for_model(
    model_key: str,
    benign_traces: List[Dict],
    backdoor_traces: List[Dict],
    extractor: FeatureExtractor
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Extract features for benign and backdoor traces"""

    print(f"\n  Extracting features for {model_key}...")
    print(f"    Benign traces: {len(benign_traces)}")
    print(f"    Backdoor traces: {len(backdoor_traces)}")

    # Extract benign features
    benign_features = []
    benign_ids = []
    for trace in benign_traces:
        features = extractor.extract(trace)
        benign_features.append([features[name] for name in extractor.get_feature_names()])
        benign_ids.append(trace['trace_id'])

    # Extract backdoor features
    backdoor_features = []
    backdoor_ids = []
    for trace in backdoor_traces:
        features = extractor.extract(trace)
        backdoor_features.append([features[name] for name in extractor.get_feature_names()])
        backdoor_ids.append(trace['trace_id'])

    return (
        np.array(benign_features),
        np.array(backdoor_features),
        benign_ids,
        backdoor_ids
    )


# =============================================================================
# Statistical Analysis
# =============================================================================

def compute_feature_statistics(
    features: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Compute statistics for each feature"""

    stats = []
    for i, name in enumerate(feature_names):
        values = features[:, i]
        stats.append({
            'feature': name,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        })

    return pd.DataFrame(stats)


def compare_distributions(
    benign_features: np.ndarray,
    backdoor_features: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Compare benign vs backdoor feature distributions"""

    comparisons = []
    for i, name in enumerate(feature_names):
        benign_vals = benign_features[:, i]
        backdoor_vals = backdoor_features[:, i]

        # Effect size (Cohen's d)
        mean_diff = np.mean(backdoor_vals) - np.mean(benign_vals)
        pooled_std = np.sqrt((np.var(benign_vals) + np.var(backdoor_vals)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        comparisons.append({
            'feature': name,
            'benign_mean': np.mean(benign_vals),
            'backdoor_mean': np.mean(backdoor_vals),
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'abs_effect': abs(cohens_d)
        })

    df = pd.DataFrame(comparisons)
    df = df.sort_values('abs_effect', ascending=False)
    return df


# =============================================================================
# Cross-LLM Analysis
# =============================================================================

def analyze_cross_llm_patterns(
    model_features: Dict[str, Dict[str, np.ndarray]],
    feature_names: List[str]
) -> pd.DataFrame:
    """Analyze feature consistency across models"""

    print("\n" + "="*70)
    print("Cross-LLM Feature Analysis")
    print("="*70)

    # For each feature, compute variance across models
    feature_consistency = []

    for i, name in enumerate(feature_names):
        model_means = []
        for model_key in model_features:
            benign = model_features[model_key]['benign']
            if len(benign) > 0:
                model_means.append(np.mean(benign[:, i]))

        if len(model_means) > 0:
            cross_model_std = np.std(model_means)
            cross_model_cv = cross_model_std / np.mean(model_means) if np.mean(model_means) > 0 else 0

            feature_consistency.append({
                'feature': name,
                'cross_model_mean': np.mean(model_means),
                'cross_model_std': cross_model_std,
                'cross_model_cv': cross_model_cv,  # Coefficient of variation
                'stability': 'high' if cross_model_cv < 0.2 else 'medium' if cross_model_cv < 0.5 else 'low'
            })

    df = pd.DataFrame(feature_consistency)
    df = df.sort_values('cross_model_cv')
    return df


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Multi-LLM Feature Distribution Analysis")
    print("="*70)

    # Configuration
    base_dir = Path("data/multi_llm_traces")
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["gpt51", "claude45", "grok41", "llama4", "gptoss", "deepseek"]
    model_names = {
        "gpt51": "GPT-5.1",
        "claude45": "Claude Sonnet 4.5",
        "grok41": "Grok 4.1 Fast",
        "llama4": "Llama 4 Maverick",
        "gptoss": "GPT-OSS 120B",
        "deepseek": "DeepSeek Chat V3.1"
    }

    # Initialize feature extractor
    print("\n📊 Initializing feature extractor...")
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    print(f"   Total features: {len(feature_names)}")

    # Extract features for all models
    print("\n" + "="*70)
    print("Extracting Features")
    print("="*70)

    model_features = {}
    model_ids = {}

    for model_key in models:
        print(f"\n🔍 Processing {model_names[model_key]}...")

        # Load traces
        benign_traces, backdoor_traces = load_traces_for_model(model_key, base_dir)

        # Extract features
        benign_feats, backdoor_feats, benign_ids, backdoor_ids = extract_features_for_model(
            model_key, benign_traces, backdoor_traces, extractor
        )

        model_features[model_key] = {
            'benign': benign_feats,
            'backdoor': backdoor_feats
        }

        model_ids[model_key] = {
            'benign': benign_ids,
            'backdoor': backdoor_ids
        }

        print(f"    ✅ Extracted: {len(benign_feats)} benign, {len(backdoor_feats)} backdoor")

    # Per-model statistics
    print("\n" + "="*70)
    print("Per-Model Feature Statistics")
    print("="*70)

    for model_key in models:
        print(f"\n📊 {model_names[model_key]}")

        benign = model_features[model_key]['benign']
        backdoor = model_features[model_key]['backdoor']

        # Compute statistics
        benign_stats = compute_feature_statistics(benign, feature_names)
        backdoor_stats = compute_feature_statistics(backdoor, feature_names)

        # Compare distributions
        comparison = compare_distributions(benign, backdoor, feature_names)

        # Save
        benign_stats.to_csv(output_dir / f"{model_key}_benign_stats.csv", index=False)
        backdoor_stats.to_csv(output_dir / f"{model_key}_backdoor_stats.csv", index=False)
        comparison.to_csv(output_dir / f"{model_key}_comparison.csv", index=False)

        # Print top discriminative features
        print(f"\n  Top 5 discriminative features (by effect size):")
        for idx, row in comparison.head(5).iterrows():
            print(f"    {row['feature']:40} | Cohen's d: {row['cohens_d']:6.3f}")

    # Cross-LLM analysis
    consistency = analyze_cross_llm_patterns(model_features, feature_names)
    consistency.to_csv(output_dir / "cross_llm_consistency.csv", index=False)

    print("\n  Features with highest stability across models:")
    for idx, row in consistency.head(10).iterrows():
        print(f"    {row['feature']:40} | CV: {row['cross_model_cv']:.3f} ({row['stability']})")

    print("\n  Features with lowest stability (model-specific patterns):")
    for idx, row in consistency.tail(10).iterrows():
        print(f"    {row['feature']:40} | CV: {row['cross_model_cv']:.3f} ({row['stability']})")

    # Save feature matrices for later use
    print("\n" + "="*70)
    print("Saving Feature Matrices")
    print("="*70)

    for model_key in models:
        np.save(output_dir / f"{model_key}_benign_features.npy", model_features[model_key]['benign'])
        np.save(output_dir / f"{model_key}_backdoor_features.npy", model_features[model_key]['backdoor'])

        with open(output_dir / f"{model_key}_ids.json", 'w') as f:
            json.dump(model_ids[model_key], f, indent=2)

        print(f"  ✅ Saved {model_names[model_key]} features")

    # Save metadata
    metadata = {
        'extraction_date': datetime.utcnow().isoformat(),
        'models': models,
        'model_names': model_names,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'dataset_sizes': {
            model_key: {
                'benign': len(model_features[model_key]['benign']),
                'backdoor': len(model_features[model_key]['backdoor'])
            }
            for model_key in models
        }
    }

    with open(output_dir / "extraction_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - Per-model statistics: *_benign_stats.csv, *_backdoor_stats.csv")
    print(f"  - Per-model comparisons: *_comparison.csv")
    print(f"  - Cross-LLM consistency: cross_llm_consistency.csv")
    print(f"  - Feature matrices: *_benign_features.npy, *_backdoor_features.npy")
    print(f"  - Metadata: extraction_metadata.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
