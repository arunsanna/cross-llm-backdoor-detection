#!/usr/bin/env python3
"""
Cross-LLM Detection Matrix
Train detector on one model, test on all others
Generates 6x6 detection matrix for paper

Usage:
    python cross_llm_detection_matrix.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Data Loading
# =============================================================================

def load_model_features(model_key: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load benign and backdoor features for a model"""
    benign = np.load(data_dir / f"{model_key}_benign_features.npy")
    backdoor = np.load(data_dir / f"{model_key}_backdoor_features.npy")
    return benign, backdoor


def create_dataset(benign: np.ndarray, backdoor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create X, y dataset from benign/backdoor features"""
    X = np.vstack([benign, backdoor])
    y = np.array([0] * len(benign) + [1] * len(backdoor))
    return X, y


# =============================================================================
# Detection Evaluation
# =============================================================================

def train_and_evaluate(
    train_benign: np.ndarray,
    train_backdoor: np.ndarray,
    test_benign: np.ndarray,
    test_backdoor: np.ndarray
) -> Dict[str, float]:
    """Train detector on train set, evaluate on test set"""

    # Create datasets
    X_train, y_train = create_dataset(train_benign, train_backdoor)
    X_test, y_test = create_dataset(test_benign, test_backdoor)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba)
    }

    return metrics


# =============================================================================
# Cross-LLM Matrix Generation
# =============================================================================

def generate_detection_matrix(
    models: list,
    model_features: Dict[str, Dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Generate full 6x6 detection matrix"""

    print("\n" + "="*70)
    print("Generating Cross-LLM Detection Matrix")
    print("="*70)
    print("\nTraining detector on each model, testing on all models...")
    print(f"Total experiments: {len(models) * len(models)} (6x6 matrix)\n")

    results = []

    for i, train_model in enumerate(models, 1):
        train_benign = model_features[train_model]['benign']
        train_backdoor = model_features[train_model]['backdoor']

        print(f"[{i}/6] Training on {train_model:15} | ", end='', flush=True)

        row_results = []

        for test_model in models:
            test_benign = model_features[test_model]['benign']
            test_backdoor = model_features[test_model]['backdoor']

            # Train and evaluate
            metrics = train_and_evaluate(
                train_benign, train_backdoor,
                test_benign, test_backdoor
            )

            row_results.append(metrics['accuracy'])

            results.append({
                'train_model': train_model,
                'test_model': test_model,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc']
            })

        # Print row summary
        same_model_acc = row_results[models.index(train_model)]
        avg_other_acc = np.mean([acc for j, acc in enumerate(row_results) if models[j] != train_model])
        print(f"Same: {same_model_acc:.1%} | Avg Others: {avg_other_acc:.1%}")

    return pd.DataFrame(results)


def create_heatmap_matrix(results_df: pd.DataFrame, models: list, metric: str = 'accuracy') -> pd.DataFrame:
    """Create NxN matrix for heatmap visualization"""
    matrix = np.zeros((len(models), len(models)))

    for i, train_model in enumerate(models):
        for j, test_model in enumerate(models):
            value = results_df[
                (results_df['train_model'] == train_model) &
                (results_df['test_model'] == test_model)
            ][metric].values[0]
            matrix[i, j] = value

    return pd.DataFrame(matrix, index=models, columns=models)


# =============================================================================
# Analysis
# =============================================================================

def analyze_generalization_patterns(results_df: pd.DataFrame, models: list, model_names: dict):
    """Analyze cross-LLM generalization patterns"""

    print("\n" + "="*70)
    print("Cross-LLM Generalization Analysis")
    print("="*70)

    # Same-model performance (diagonal)
    print("\n1. Same-Model Detection (Train=Test):")
    for model in models:
        same_model = results_df[
            (results_df['train_model'] == model) &
            (results_df['test_model'] == model)
        ].iloc[0]
        print(f"   {model_names[model]:25} | Acc: {same_model['accuracy']:.1%} | "
              f"F1: {same_model['f1']:.1%} | AUC: {same_model['auc']:.3f}")

    # Cross-model performance (off-diagonal)
    print("\n2. Cross-Model Generalization (Train≠Test):")
    for model in models:
        cross_model = results_df[
            (results_df['train_model'] == model) &
            (results_df['test_model'] != model)
        ]
        avg_acc = cross_model['accuracy'].mean()
        avg_f1 = cross_model['f1'].mean()
        print(f"   {model_names[model]:25} → Others | Avg Acc: {avg_acc:.1%} | Avg F1: {avg_f1:.1%}")

    # Overall statistics
    print("\n3. Overall Statistics:")
    same_model_avg = results_df[results_df['train_model'] == results_df['test_model']]['accuracy'].mean()
    cross_model_avg = results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].mean()
    generalization_gap = same_model_avg - cross_model_avg

    print(f"   Same-model avg:        {same_model_avg:.1%}")
    print(f"   Cross-model avg:       {cross_model_avg:.1%}")
    print(f"   Generalization gap:    {generalization_gap:.1%}")
    print(f"   Generalization ratio:  {cross_model_avg/same_model_avg:.1%}")

    # Best/worst transfer pairs
    print("\n4. Best Transfer Pairs (Train→Test):")
    cross_only = results_df[results_df['train_model'] != results_df['test_model']].copy()
    cross_only = cross_only.sort_values('accuracy', ascending=False)

    for idx, row in cross_only.head(5).iterrows():
        print(f"   {model_names[row['train_model']]:20} → {model_names[row['test_model']]:20} | "
              f"Acc: {row['accuracy']:.1%} | F1: {row['f1']:.1%}")

    print("\n5. Worst Transfer Pairs (Train→Test):")
    for idx, row in cross_only.tail(5).iterrows():
        print(f"   {model_names[row['train_model']]:20} → {model_names[row['test_model']]:20} | "
              f"Acc: {row['accuracy']:.1%} | F1: {row['f1']:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("Cross-LLM Detection Matrix Generation")
    print("="*70)

    # Configuration
    data_dir = Path("data/analysis")
    output_dir = Path("data/cross_llm_results")
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

    # Load all feature matrices
    print("\n📊 Loading feature matrices...")
    model_features = {}

    for model in models:
        benign, backdoor = load_model_features(model, data_dir)
        model_features[model] = {
            'benign': benign,
            'backdoor': backdoor
        }
        print(f"   ✅ {model_names[model]:25} | Benign: {len(benign)} | Backdoor: {len(backdoor)}")

    # Generate detection matrix
    results_df = generate_detection_matrix(models, model_features)

    # Save detailed results
    results_df.to_csv(output_dir / "detection_matrix_detailed.csv", index=False)
    print(f"\n✅ Saved detailed results: detection_matrix_detailed.csv")

    # Create heatmap matrices
    for metric in ['accuracy', 'f1', 'auc']:
        heatmap = create_heatmap_matrix(results_df, models, metric)
        heatmap.to_csv(output_dir / f"detection_matrix_{metric}.csv")
        print(f"✅ Saved {metric} heatmap: detection_matrix_{metric}.csv")

    # Analysis
    analyze_generalization_patterns(results_df, models, model_names)

    # Save summary statistics
    summary = {
        'same_model_performance': results_df[results_df['train_model'] == results_df['test_model']][
            ['train_model', 'accuracy', 'f1', 'auc']
        ].to_dict('records'),
        'cross_model_statistics': {
            'mean_accuracy': float(results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].mean()),
            'std_accuracy': float(results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].std()),
            'min_accuracy': float(results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].min()),
            'max_accuracy': float(results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].max())
        }
    }

    with open(output_dir / "detection_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved summary: detection_summary.json")

    print("\n" + "="*70)
    print("✅ Cross-LLM Detection Matrix Complete!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey findings:")
    same_avg = results_df[results_df['train_model'] == results_df['test_model']]['accuracy'].mean()
    cross_avg = results_df[results_df['train_model'] != results_df['test_model']]['accuracy'].mean()
    print(f"  - Same-model detection:  {same_avg:.1%}")
    print(f"  - Cross-model detection: {cross_avg:.1%}")
    print(f"  - Generalization gap:    {same_avg - cross_avg:.1%}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
