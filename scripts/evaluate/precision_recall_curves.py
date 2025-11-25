"""
Generate Precision-Recall Curves for Threshold Tuning

Shows the trade-off between precision and recall at different thresholds,
demonstrating the detector's flexibility for different deployment scenarios.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector


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


def plot_precision_recall_curve(y_test, y_proba, model_name, save_path):
    """Plot precision-recall curve for a single model."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AP = {ap:.3f})')

    # Mark some interesting operating points
    # High precision (conservative): 90% precision
    high_prec_idx = np.where(precision >= 0.90)[0]
    if len(high_prec_idx) > 0:
        idx = high_prec_idx[0]
        plt.plot(recall[idx], precision[idx], 'ro', markersize=10,
                label=f'High Precision: P={precision[idx]:.2f}, R={recall[idx]:.2f}')

    # Balanced: F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    plt.plot(recall[best_f1_idx], precision[best_f1_idx], 'go', markersize=10,
            label=f'Best F1: P={precision[best_f1_idx]:.2f}, R={recall[best_f1_idx]:.2f}')

    # High recall (aggressive): 90% recall
    high_rec_idx = np.where(recall >= 0.90)[0]
    if len(high_rec_idx) > 0:
        idx = high_rec_idx[-1]
        plt.plot(recall[idx], precision[idx], 'bo', markersize=10,
                label=f'High Recall: P={precision[idx]:.2f}, R={recall[idx]:.2f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Backdoor Detection Rate)', fontsize=12)
    plt.ylabel('Precision (Backdoor Prediction Accuracy)', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    return ap, precision, recall, thresholds


def main():
    print("\n" + "="*70)
    print("PRECISION-RECALL CURVES FOR THRESHOLD TUNING")
    print("="*70 + "\n")

    # Setup
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Dataset: {len(X)} traces ({np.sum(y==0)} clean, {np.sum(y==1)} backdoor)")
    print(f"Test set: {len(X_test)} traces\n")

    # Train models and generate curves
    results = {}

    for model_type in ['svm', 'random_forest']:
        print(f"Training {model_type.upper()}...")
        detector = AnomalyDetector(model_type=model_type)
        detector.fit(X_train, y_train, feature_names=extractor.get_feature_names())

        y_proba = detector.predict_proba(X_test)

        model_name = model_type.replace('_', ' ').title()
        save_path = output_dir / f"precision_recall_{model_type}.png"

        ap, precision, recall, thresholds = plot_precision_recall_curve(
            y_test, y_proba, model_name, save_path
        )

        results[model_type] = {
            'average_precision': float(ap),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist()
        }

        print(f"  Average Precision: {ap:.3f}\n")

    # Combined plot
    print("Generating combined plot...")
    plt.figure(figsize=(10, 8))

    colors = {'svm': '#e74c3c', 'random_forest': '#3498db'}

    for model_type, data in results.items():
        model_name = model_type.replace('_', ' ').title()
        plt.plot(data['recall'], data['precision'],
                linewidth=2, color=colors[model_type],
                label=f'{model_name} (AP = {data["average_precision"]:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Backdoor Detection Rate)', fontsize=12)
    plt.ylabel('Precision (Backdoor Prediction Accuracy)', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combined_path = output_dir / "precision_recall_comparison.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {combined_path}")
    plt.close()

    # Save results
    output_file = Path("precision_recall_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70 + "\n")

    print("Operating Points for Different Deployment Scenarios:")
    print("-" * 70)
    print("1. High Precision (Security-Critical):")
    print("   - Use when false positives are very costly")
    print("   - Example: ~90% precision, moderate recall")
    print("   - Suitable for: Production CI/CD pipelines")
    print()
    print("2. Balanced (General Purpose):")
    print("   - Best F1 score")
    print("   - Example: ~80% precision, ~80% recall")
    print("   - Suitable for: Pre-deployment scanning")
    print()
    print("3. High Recall (Threat Hunting):")
    print("   - Use when missing backdoors is very costly")
    print("   - Example: ~60% precision, ~90% recall")
    print("   - Suitable for: Security audits, incident response")
    print()

    print("Figures saved:")
    print(f"  - {output_dir}/precision_recall_svm.png")
    print(f"  - {output_dir}/precision_recall_random_forest.png")
    print(f"  - {output_dir}/precision_recall_comparison.png")
    print()


if __name__ == "__main__":
    main()
