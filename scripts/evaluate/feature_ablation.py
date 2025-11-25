"""
Feature Ablation Study

Tests which feature categories are most important for detection:
- Temporal features only
- Structural features only
- Action features only
- Data features only
- All features (baseline)
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector, ModelEvaluator


def get_feature_categories(feature_names):
    """Categorize features by type."""
    categories = {
        'temporal': [],
        'structural': [],
        'action': [],
        'data': []
    }

    for i, name in enumerate(feature_names):
        if any(keyword in name for keyword in ['time', 'duration', 'latency', 'interval', 'timestamp']):
            categories['temporal'].append(i)
        elif any(keyword in name for keyword in ['depth', 'length', 'count', 'steps', 'ratio']):
            categories['structural'].append(i)
        elif any(keyword in name for keyword in ['tool', 'action', 'call', 'type']):
            categories['action'].append(i)
        elif any(keyword in name for keyword in ['data', 'size', 'input', 'output']):
            categories['data'].append(i)

    return categories


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


def evaluate_feature_subset(X_train, y_train, X_test, y_test, feature_indices, subset_name):
    """Train and evaluate model with only a subset of features."""
    # Extract only the specified features
    X_train_subset = X_train[:, feature_indices]
    X_test_subset = X_test[:, feature_indices]

    # Train SVM on subset
    detector = AnomalyDetector(model_type='svm')
    detector.model.fit(X_train_subset, y_train)

    # Evaluate
    y_pred = detector.model.predict(X_test_subset)
    y_proba = detector.model.decision_function(X_test_subset)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)

    print(f"{subset_name}:")
    print(f"  Features: {len(feature_indices)}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print()

    return metrics


def main():
    print("\n" + "="*70)
    print("FEATURE ABLATION STUDY")
    print("="*70 + "\n")

    # Initialize
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()

    # Load data
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")
    backdoor_dirs = [
        (Path("data/backdoor_traces/TM1"), 1),
        (Path("data/backdoor_traces/TM2"), 1),
        (Path("data/backdoor_traces/TM3"), 1),
    ]

    print("Loading traces...")
    X, y = load_traces(clean_dir, backdoor_dirs, extractor, max_clean=600)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Dataset: {len(X)} traces ({np.sum(y==0)} clean, {np.sum(y==1)} backdoor)")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    # Categorize features
    categories = get_feature_categories(feature_names)

    print("Feature Categories:")
    for cat, indices in categories.items():
        print(f"  {cat.capitalize()}: {len(indices)} features")
    print()

    # Run ablation experiments
    results = {}

    print("="*70)
    print("ABLATION EXPERIMENTS (SVM)")
    print("="*70 + "\n")

    # Baseline: All features
    all_indices = list(range(len(feature_names)))
    results['all'] = evaluate_feature_subset(X_train, y_train, X_test, y_test, all_indices, "Baseline (All Features)")

    # Temporal only
    if len(categories['temporal']) > 0:
        results['temporal'] = evaluate_feature_subset(X_train, y_train, X_test, y_test,
                                                      categories['temporal'], "Temporal Features Only")

    # Structural only
    if len(categories['structural']) > 0:
        results['structural'] = evaluate_feature_subset(X_train, y_train, X_test, y_test,
                                                        categories['structural'], "Structural Features Only")

    # Action only
    if len(categories['action']) > 0:
        results['action'] = evaluate_feature_subset(X_train, y_train, X_test, y_test,
                                                    categories['action'], "Action Features Only")

    # Data only
    if len(categories['data']) > 0:
        results['data'] = evaluate_feature_subset(X_train, y_train, X_test, y_test,
                                                  categories['data'], "Data Features Only")

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70 + "\n")

    print("Recall Performance (most important for backdoor detection):")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True)
    for subset, metrics in sorted_results:
        recall_drop = (results['all']['recall'] - metrics['recall']) * 100
        print(f"  {subset.capitalize():<20} {metrics['recall']:.3f} ({recall_drop:+.1f}% vs baseline)")
    print()

    # Find most critical category
    max_recall = max(metrics['recall'] for subset, metrics in results.items() if subset != 'all')
    critical_subset = [subset for subset, metrics in results.items()
                      if subset != 'all' and metrics['recall'] == max_recall][0]

    print(f"Most Critical Feature Category: {critical_subset.upper()}")
    print(f"  Achieves {results[critical_subset]['recall']:.1%} recall with only {len(categories[critical_subset])} features")
    print(f"  vs {results['all']['recall']:.1%} with all {len(feature_names)} features")
    print()

    # Save results
    output_file = Path("feature_ablation_results.json")
    output_data = {
        subset: {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        for subset, metrics in results.items()
    }
    output_data['feature_counts'] = {cat: len(indices) for cat, indices in categories.items()}

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
