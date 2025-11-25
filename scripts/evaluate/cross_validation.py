"""
5-Fold Cross-Validation

Validates that results generalize across different data splits.
Standard requirement for ML venues (NeurIPS, ICML).
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector, ModelEvaluator


def load_traces(clean_dir, backdoor_dirs, extractor, max_clean=600):
    """Load and extract features from traces."""
    all_features = []
    all_labels = []

    # Load clean traces
    clean_files = sorted(clean_dir.glob("*.json"))
    if len(clean_files) > max_clean:
        np.random.seed(42)
        clean_files = np.random.choice(clean_files, max_clean, replace=False)

    print(f"Loading {len(clean_files)} clean traces...")
    for trace_file in clean_files:
        try:
            with open(trace_file) as f:
                trace = json.load(f)
            features = extractor.extract(trace)
            feature_vector = [features.get(name, 0.0) for name in extractor.get_feature_names()]
            all_features.append(feature_vector)
            all_labels.append(0)
        except:
            continue

    # Load backdoor traces
    for backdoor_dir, _ in backdoor_dirs:
        backdoor_files = sorted(backdoor_dir.glob("*.json"))
        print(f"Loading {len(backdoor_files)} backdoor traces from {backdoor_dir.name}...")
        for trace_file in backdoor_files:
            try:
                with open(trace_file) as f:
                    trace = json.load(f)
                features = extractor.extract(trace)
                feature_vector = [features.get(name, 0.0) for name in extractor.get_feature_names()]
                all_features.append(feature_vector)
                all_labels.append(1)
            except:
                continue

    return np.array(all_features), np.array(all_labels)


def cross_validate_model(X, y, model_type, n_splits=5):
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    evaluator = ModelEvaluator()

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_splits}:")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Train: {len(X_train)} traces ({np.sum(y_train==0)} clean, {np.sum(y_train==1)} backdoor)")
        print(f"  Test:  {len(X_test)} traces ({np.sum(y_test==0)} clean, {np.sum(y_test==1)} backdoor)")

        # Train model
        detector = AnomalyDetector(model_type=model_type)
        detector.fit(X_train, y_train, feature_names=None)

        # Evaluate
        y_pred = detector.predict(X_test)
        y_proba = detector.predict_proba(X_test)

        metrics = evaluator.evaluate(y_test, y_pred, y_proba)

        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")

        fold_results.append(metrics)

    return fold_results


def aggregate_results(fold_results):
    """Compute mean and std across folds."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    aggregated = {}

    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }

    return aggregated


def main():
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION")
    print("="*70 + "\n")

    # Initialize
    extractor = FeatureExtractor()

    # Load data
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")
    backdoor_dirs = [
        (Path("data/backdoor_traces/TM1"), 1),
        (Path("data/backdoor_traces/TM2"), 1),
        (Path("data/backdoor_traces/TM3"), 1),
    ]

    print("Loading traces...")
    X, y = load_traces(clean_dir, backdoor_dirs, extractor, max_clean=600)

    print(f"\nDataset: {len(X)} traces ({np.sum(y==0)} clean, {np.sum(y==1)} backdoor)\n")

    # Cross-validate each model
    results = {}

    for model_type in ['svm', 'random_forest']:
        print("="*70)
        print(f"{model_type.upper()} - 5-Fold Cross-Validation")
        print("="*70)

        fold_results = cross_validate_model(X, y, model_type, n_splits=5)
        results[model_type] = aggregate_results(fold_results)

    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70 + "\n")

    for model_type in ['svm', 'random_forest']:
        print(f"{model_type.upper()}:")
        agg = results[model_type]

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean = agg[metric]['mean']
            std = agg[metric]['std']
            min_val = agg[metric]['min']
            max_val = agg[metric]['max']
            print(f"  {metric.capitalize():<12} {mean:.3f} ± {std:.3f}  (min: {min_val:.3f}, max: {max_val:.3f})")
        print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70 + "\n")

    svm_recall_mean = results['svm']['recall']['mean']
    svm_recall_std = results['svm']['recall']['std']
    rf_recall_mean = results['random_forest']['recall']['mean']
    rf_recall_std = results['random_forest']['recall']['std']

    print(f"SVM Recall:           {svm_recall_mean:.1%} ± {svm_recall_std:.1%}")
    print(f"Random Forest Recall: {rf_recall_mean:.1%} ± {rf_recall_std:.1%}")
    print()

    print(f"Variance Analysis:")
    print(f"  SVM std:           {svm_recall_std:.3f} ({'low' if svm_recall_std < 0.05 else 'medium' if svm_recall_std < 0.10 else 'high'} variance)")
    print(f"  Random Forest std: {rf_recall_std:.3f} ({'low' if rf_recall_std < 0.05 else 'medium' if rf_recall_std < 0.10 else 'high'} variance)")
    print()

    if svm_recall_std < 0.10 and rf_recall_std < 0.10:
        print("✅ Results are stable across folds (low variance)")
    elif svm_recall_std < 0.15 and rf_recall_std < 0.15:
        print("⚠️  Results show moderate variance across folds")
    else:
        print("❌ Results show high variance across folds - may need more data")
    print()

    # Save results
    output_file = Path("cross_validation_results.json")
    output_data = {
        model_type: {
            metric: {k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in stats.items() if k != 'values'}
            for metric, stats in agg.items()
        }
        for model_type, agg in results.items()
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
