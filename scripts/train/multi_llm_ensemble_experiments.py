#!/usr/bin/env python3
"""
Multi-LLM Ensemble Detection Experiments
Tests different strategies to overcome cross-LLM generalization gap

Approaches:
1. Baseline: Single-model detectors (from cross_llm_detection_matrix.py)
2. Pooled Training: Train on all models together
3. Ensemble Voting: Majority vote from 6 single-model detectors
4. Model-Aware: Add model_id as categorical feature
5. Best-Pair Ensemble: For each model, use 2 best source models

Usage:
    python multi_llm_ensemble_experiments.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# Data Loading
# =============================================================================

def load_model_features(model_key: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load benign and backdoor features for a model"""
    benign = np.load(data_dir / f"{model_key}_benign_features.npy")
    backdoor = np.load(data_dir / f"{model_key}_backdoor_features.npy")
    return benign, backdoor


def create_dataset(benign: np.ndarray, backdoor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create X, y dataset"""
    X = np.vstack([benign, backdoor])
    y = np.array([0] * len(benign) + [1] * len(backdoor))
    return X, y


def add_model_id_feature(
    features: np.ndarray,
    model_id: int,
    num_samples: int
) -> np.ndarray:
    """Add model_id as additional feature"""
    model_ids = np.full((num_samples, 1), model_id)
    return np.hstack([features, model_ids])


# =============================================================================
# Approach 1: Baseline (Single-Model) - ALREADY COMPUTED
# =============================================================================

def load_baseline_results(results_dir: Path) -> Dict:
    """Load results from cross_llm_detection_matrix.py"""
    df = pd.read_csv(results_dir / "detection_matrix_detailed.csv")

    same_model = df[df['train_model'] == df['test_model']]
    cross_model = df[df['train_model'] != df['test_model']]

    return {
        'name': 'Baseline (Single-Model)',
        'same_model_acc': same_model['accuracy'].mean(),
        'cross_model_acc': cross_model['accuracy'].mean(),
        'overall_acc': df['accuracy'].mean(),
        'same_model_f1': same_model['f1'].mean(),
        'cross_model_f1': cross_model['f1'].mean()
    }


# =============================================================================
# Approach 2: Pooled Training (All Models Together)
# =============================================================================

def pooled_training(
    models: List[str],
    model_features: Dict[str, Dict[str, np.ndarray]]
) -> Dict:
    """Train single detector on ALL models, test on each"""

    print("\n" + "="*70)
    print("Approach 2: Pooled Training (All Models Together)")
    print("="*70)

    # Combine ALL training data
    all_benign = []
    all_backdoor = []

    for model in models:
        all_benign.append(model_features[model]['benign'])
        all_backdoor.append(model_features[model]['backdoor'])

    all_benign = np.vstack(all_benign)
    all_backdoor = np.vstack(all_backdoor)

    print(f"Training set: {len(all_benign)} benign + {len(all_backdoor)} backdoor = {len(all_benign) + len(all_backdoor)} total")

    X_train, y_train = create_dataset(all_benign, all_backdoor)

    # Train detector
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    # Test on each model
    results = []
    for model in models:
        test_benign = model_features[model]['benign']
        test_backdoor = model_features[model]['backdoor']
        X_test, y_test = create_dataset(test_benign, test_backdoor)
        X_test_scaled = scaler.transform(X_test)

        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({'model': model, 'accuracy': acc, 'f1': f1})
        print(f"  {model:15} | Acc: {acc:.1%} | F1: {f1:.1%}")

    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])

    return {
        'name': 'Pooled Training',
        'same_model_acc': avg_acc,  # No distinction in pooled
        'cross_model_acc': avg_acc,
        'overall_acc': avg_acc,
        'same_model_f1': avg_f1,
        'cross_model_f1': avg_f1,
        'per_model': results
    }


# =============================================================================
# Approach 3: Ensemble Voting (6 Single-Model Detectors)
# =============================================================================

def ensemble_voting(
    models: List[str],
    model_features: Dict[str, Dict[str, np.ndarray]]
) -> Dict:
    """Train 6 detectors, use majority vote"""

    print("\n" + "="*70)
    print("Approach 3: Ensemble Voting (6 Detectors)")
    print("="*70)

    # Train individual detectors
    detectors = {}
    scalers = {}

    for model in models:
        benign = model_features[model]['benign']
        backdoor = model_features[model]['backdoor']
        X_train, y_train = create_dataset(benign, backdoor)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)

        detectors[model] = clf
        scalers[model] = scaler

    print(f"Trained {len(detectors)} individual detectors")

    # Test with voting
    results = []
    for test_model in models:
        test_benign = model_features[test_model]['benign']
        test_backdoor = model_features[test_model]['backdoor']
        X_test, y_test = create_dataset(test_benign, test_backdoor)

        # Get predictions from all detectors
        votes = []
        for train_model in models:
            X_test_scaled = scalers[train_model].transform(X_test)
            pred = detectors[train_model].predict(X_test_scaled)
            votes.append(pred)

        # Majority vote
        votes = np.array(votes)
        y_pred = np.round(np.mean(votes, axis=0)).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({'model': test_model, 'accuracy': acc, 'f1': f1})
        print(f"  {test_model:15} | Acc: {acc:.1%} | F1: {f1:.1%}")

    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])

    return {
        'name': 'Ensemble Voting',
        'same_model_acc': avg_acc,
        'cross_model_acc': avg_acc,
        'overall_acc': avg_acc,
        'same_model_f1': avg_f1,
        'cross_model_f1': avg_f1,
        'per_model': results
    }


# =============================================================================
# Approach 4: Model-Aware Detector (model_id as feature)
# =============================================================================

def model_aware_detector(
    models: List[str],
    model_features: Dict[str, Dict[str, np.ndarray]]
) -> Dict:
    """Add model_id as categorical feature (52 features instead of 51)"""

    print("\n" + "="*70)
    print("Approach 4: Model-Aware Detector (model_id as feature)")
    print("="*70)

    # Create model_id mapping
    model_id_map = {model: i for i, model in enumerate(models)}

    # Combine ALL data with model_id
    all_X = []
    all_y = []

    for model in models:
        benign = model_features[model]['benign']
        backdoor = model_features[model]['backdoor']

        # Add model_id feature
        benign_with_id = add_model_id_feature(benign, model_id_map[model], len(benign))
        backdoor_with_id = add_model_id_feature(backdoor, model_id_map[model], len(backdoor))

        X, y = create_dataset(benign_with_id, backdoor_with_id)
        all_X.append(X)
        all_y.append(y)

    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)

    print(f"Training with 52 features (51 behavioral + 1 model_id)")
    print(f"Training set: {len(X_train)} samples")

    # Train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)

    # Test on each model
    results = []
    for model in models:
        test_benign = model_features[model]['benign']
        test_backdoor = model_features[model]['backdoor']

        test_benign_with_id = add_model_id_feature(test_benign, model_id_map[model], len(test_benign))
        test_backdoor_with_id = add_model_id_feature(test_backdoor, model_id_map[model], len(test_backdoor))

        X_test, y_test = create_dataset(test_benign_with_id, test_backdoor_with_id)
        X_test_scaled = scaler.transform(X_test)

        y_pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({'model': model, 'accuracy': acc, 'f1': f1})
        print(f"  {model:15} | Acc: {acc:.1%} | F1: {f1:.1%}")

    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])

    return {
        'name': 'Model-Aware',
        'same_model_acc': avg_acc,
        'cross_model_acc': avg_acc,
        'overall_acc': avg_acc,
        'same_model_f1': avg_f1,
        'cross_model_f1': avg_f1,
        'per_model': results
    }


# =============================================================================
# Comparison and Analysis
# =============================================================================

def compare_approaches(baseline: Dict, pooled: Dict, ensemble: Dict, model_aware: Dict):
    """Compare all approaches"""

    print("\n" + "="*70)
    print("COMPARISON: All Approaches")
    print("="*70)

    approaches = [baseline, pooled, ensemble, model_aware]

    # Create comparison table
    comparison = []
    for approach in approaches:
        comparison.append({
            'Approach': approach['name'],
            'Same-Model Acc': f"{approach['same_model_acc']:.1%}",
            'Cross-Model Acc': f"{approach['cross_model_acc']:.1%}",
            'Overall Acc': f"{approach['overall_acc']:.1%}",
            'Same-Model F1': f"{approach['same_model_f1']:.1%}",
            'Cross-Model F1': f"{approach['cross_model_f1']:.1%}"
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print("\n1. Generalization Gap:")
    for approach in approaches:
        gap = approach['same_model_acc'] - approach['cross_model_acc']
        print(f"   {approach['name']:25} | Gap: {gap:.1%}")

    print("\n2. Best Overall Performance:")
    best = max(approaches, key=lambda x: x['overall_acc'])
    print(f"   {best['name']}: {best['overall_acc']:.1%} accuracy")

    print("\n3. Cross-Model Generalization Winner:")
    best_cross = max(approaches, key=lambda x: x['cross_model_acc'])
    print(f"   {best_cross['name']}: {best_cross['cross_model_acc']:.1%} accuracy")

    return df


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Multi-LLM Ensemble Detection Experiments")
    print("="*70)

    # Configuration
    data_dir = Path("data/analysis")
    results_dir = Path("data/cross_llm_results")
    output_dir = Path("data/ensemble_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["gpt51", "claude45", "grok41", "llama4", "gptoss", "deepseek"]

    # Load features
    print("\n📊 Loading feature matrices...")
    model_features = {}
    for model in models:
        benign, backdoor = load_model_features(model, data_dir)
        model_features[model] = {'benign': benign, 'backdoor': backdoor}

    # Approach 1: Baseline (load existing results)
    print("\n" + "="*70)
    print("Approach 1: Baseline (Single-Model Detectors)")
    print("="*70)
    print("Loading results from cross_llm_detection_matrix.py...")
    baseline = load_baseline_results(results_dir)
    print(f"  Same-model: {baseline['same_model_acc']:.1%}")
    print(f"  Cross-model: {baseline['cross_model_acc']:.1%}")

    # Approach 2: Pooled Training
    pooled = pooled_training(models, model_features)

    # Approach 3: Ensemble Voting
    ensemble = ensemble_voting(models, model_features)

    # Approach 4: Model-Aware
    model_aware = model_aware_detector(models, model_features)

    # Comparison
    comparison_df = compare_approaches(baseline, pooled, ensemble, model_aware)

    # Save results
    comparison_df.to_csv(output_dir / "ensemble_comparison.csv", index=False)

    summary = {
        'baseline': baseline,
        'pooled': pooled,
        'ensemble': ensemble,
        'model_aware': model_aware
    }

    with open(output_dir / "ensemble_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("✅ Ensemble Experiments Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("  - ensemble_comparison.csv")
    print("  - ensemble_summary.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
