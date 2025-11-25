"""
Generate visualizations for research paper.

Creates:
- ROC curves for all models
- Confusion matrices
- Feature importance plots
- Per-threat-model comparison charts
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle

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


def plot_roc_comparison(models_data, save_path):
    """Plot ROC curves for all models on same plot."""
    plt.figure(figsize=(10, 8))

    for model_name, data in models_data.items():
        fpr = data['fpr']
        tpr = data['tpr']
        auc = data['auc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name.upper()} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Backdoor Detection (All Models)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(models_data, save_path):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (model_name, data) in enumerate(models_data.items()):
        cm = data['confusion_matrix']
        ax = axes[idx]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Clean', 'Backdoor'],
                   yticklabels=['Clean', 'Backdoor'])
        ax.set_title(f'{model_name.upper()}\n(Acc: {data["accuracy"]:.3f}, Recall: {data["recall"]:.3f})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_feature_importance(feature_importance, save_path):
    """Plot top 20 most important features."""
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances, color='steelblue')
    plt.yticks(y_pos, features, fontsize=10)
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 20 Most Important Features (Random Forest)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_tm_comparison(tm_results, save_path):
    """Plot per-threat-model detection performance."""
    tms = ['TM1\n(Data Poisoning)', 'TM2\n(Tool Manipulation)', 'TM3\n(Model Tampering)', 'All TMs\nCombined']
    recalls = [tm_results['TM1']['recall'], tm_results['TM2']['recall'],
               tm_results['TM3']['recall'], tm_results['All']['recall']]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(tms, recalls, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, recall in zip(bars, recalls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{recall:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Recall (Detection Rate)', fontsize=12)
    ax.set_title('Per-Threat-Model Detection Performance (SVM)', fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    # Setup
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS FOR PAPER")
    print("="*70 + "\n")

    # Initialize feature extractor
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

    # Train models and collect data
    models_data = {}
    evaluator = ModelEvaluator()

    for model_type in ['random_forest', 'svm', 'isolation_forest']:
        print(f"Training {model_type}...")
        detector = AnomalyDetector(model_type=model_type)
        detector.fit(X_train, y_train, feature_names=extractor.get_feature_names())

        y_pred = detector.predict(X_test)
        y_proba = detector.predict_proba(X_test)

        metrics = evaluator.evaluate(y_test, y_pred, y_proba)

        # Get ROC curve data
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        models_data[model_type] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': metrics['roc_auc'],
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall']
        }

        # Save feature importance for Random Forest
        if model_type == 'random_forest':
            feature_importance = detector.get_feature_importance()

    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70 + "\n")

    # 1. ROC curves comparison
    plot_roc_comparison(models_data, output_dir / "roc_curves_comparison.png")

    # 2. Confusion matrices
    plot_confusion_matrices(models_data, output_dir / "confusion_matrices.png")

    # 3. Feature importance
    plot_feature_importance(feature_importance, output_dir / "feature_importance.png")

    # 4. Per-TM comparison
    tm_results = {
        'TM1': {'recall': 0.605},  # From previous runs
        'TM2': {'recall': 0.848},
        'TM3': {'recall': 0.045},
        'All': {'recall': 0.745}
    }
    plot_per_tm_comparison(tm_results, output_dir / "per_tm_comparison.png")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - roc_curves_comparison.png")
    print("  - confusion_matrices.png")
    print("  - feature_importance.png")
    print("  - per_tm_comparison.png")
    print()


if __name__ == "__main__":
    main()
