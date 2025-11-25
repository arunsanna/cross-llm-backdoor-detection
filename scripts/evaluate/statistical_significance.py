"""
Statistical Significance Testing via Bootstrap

Tests whether multi-TM training improvement is statistically significant
using bootstrap resampling with 1000 iterations.
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector


def bootstrap_metric(X, y, model_type, metric_fn, n_iterations=1000, test_size=0.2):
    """Bootstrap confidence intervals for a metric."""
    metrics = []

    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_boot, y_boot, test_size=test_size, random_state=i, stratify=y_boot
        )

        # Train and evaluate
        detector = AnomalyDetector(model_type=model_type)
        detector.fit(X_train, y_train)
        y_pred = detector.predict(X_test)

        # Compute metric
        metric_value = metric_fn(y_test, y_pred)
        metrics.append(metric_value)

    return np.array(metrics)


def recall_metric(y_true, y_pred):
    """Compute recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def load_data(extractor, clean_dir, backdoor_dirs, max_clean=600):
    """Load and extract features."""
    all_features = []
    all_labels = []

    # Load clean
    clean_files = sorted(clean_dir.glob("*.json"))
    if len(clean_files) > max_clean:
        np.random.seed(42)
        clean_files = np.random.choice(clean_files, max_clean, replace=False)

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

    # Load backdoors
    for backdoor_dir, _ in backdoor_dirs:
        backdoor_files = sorted(backdoor_dir.glob("*.json"))
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


def main():
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTING (Bootstrap)")
    print("="*70 + "\n")

    extractor = FeatureExtractor()
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")

    # Load data for different scenarios
    print("Loading datasets...\n")

    # All TMs combined
    all_backdoor_dirs = [
        (Path("data/backdoor_traces/TM1"), 1),
        (Path("data/backdoor_traces/TM2"), 1),
        (Path("data/backdoor_traces/TM3"), 1),
    ]
    X_all, y_all = load_data(extractor, clean_dir, all_backdoor_dirs)
    print(f"All TMs: {len(X_all)} traces ({np.sum(y_all==0)} clean, {np.sum(y_all==1)} backdoor)")

    # Individual TMs
    X_tm1, y_tm1 = load_data(extractor, clean_dir, [(Path("data/backdoor_traces/TM1"), 1)], max_clean=200)
    print(f"TM1 only: {len(X_tm1)} traces ({np.sum(y_tm1==0)} clean, {np.sum(y_tm1==1)} backdoor)")

    X_tm3, y_tm3 = load_data(extractor, clean_dir, [(Path("data/backdoor_traces/TM3"), 1)], max_clean=200)
    print(f"TM3 only: {len(X_tm3)} traces ({np.sum(y_tm3==0)} clean, {np.sum(y_tm3==1)} backdoor)")

    print("\n" + "="*70)
    print("Bootstrap Analysis (1000 iterations)")
    print("="*70 + "\n")

    # Bootstrap for all TMs combined
    print("Computing confidence intervals for All TMs (SVM)...")
    np.random.seed(42)
    recalls_all = bootstrap_metric(X_all, y_all, "svm", recall_metric, n_iterations=1000)

    ci_all = np.percentile(recalls_all, [2.5, 97.5])
    mean_all = np.mean(recalls_all)

    print(f"  Mean Recall: {mean_all:.3f}")
    print(f"  95% CI: [{ci_all[0]:.3f}, {ci_all[1]:.3f}]")
    print(f"  Std Dev: {np.std(recalls_all):.3f}\n")

    # Bootstrap for TM1 only
    print("Computing confidence intervals for TM1 only (SVM)...")
    np.random.seed(42)
    recalls_tm1 = bootstrap_metric(X_tm1, y_tm1, "svm", recall_metric, n_iterations=1000)

    ci_tm1 = np.percentile(recalls_tm1, [2.5, 97.5])
    mean_tm1 = np.mean(recalls_tm1)

    print(f"  Mean Recall: {mean_tm1:.3f}")
    print(f"  95% CI: [{ci_tm1[0]:.3f}, {ci_tm1[1]:.3f}]")
    print(f"  Std Dev: {np.std(recalls_tm1):.3f}\n")

    # Bootstrap for TM3 only
    print("Computing confidence intervals for TM3 only (SVM)...")
    np.random.seed(42)
    recalls_tm3 = bootstrap_metric(X_tm3, y_tm3, "svm", recall_metric, n_iterations=1000)

    ci_tm3 = np.percentile(recalls_tm3, [2.5, 97.5])
    mean_tm3 = np.mean(recalls_tm3)

    print(f"  Mean Recall: {mean_tm3:.3f}")
    print(f"  95% CI: [{ci_tm3[0]:.3f}, {ci_tm3[1]:.3f}]")
    print(f"  Std Dev: {np.std(recalls_tm3):.3f}\n")

    # Statistical tests
    print("="*70)
    print("Statistical Significance Tests")
    print("="*70 + "\n")

    # Mann-Whitney U test (non-parametric)
    stat_all_tm1, p_all_tm1 = stats.mannwhitneyu(recalls_all, recalls_tm1, alternative='greater')
    stat_all_tm3, p_all_tm3 = stats.mannwhitneyu(recalls_all, recalls_tm3, alternative='greater')

    print("All TMs vs TM1 only:")
    print(f"  Improvement: {mean_all - mean_tm1:.3f} ({(mean_all/mean_tm1 if mean_tm1 > 0 else float('inf')):.2f}x)")
    print(f"  Mann-Whitney U statistic: {stat_all_tm1:.2f}")
    print(f"  p-value: {p_all_tm1:.6f}")
    print(f"  Significant: {'YES ✅' if p_all_tm1 < 0.05 else 'NO ❌'}\n")

    print("All TMs vs TM3 only:")
    print(f"  Improvement: {mean_all - mean_tm3:.3f} ({(mean_all/mean_tm3 if mean_tm3 > 0 else float('inf')):.2f}x)")
    print(f"  Mann-Whitney U statistic: {stat_all_tm3:.2f}")
    print(f"  p-value: {p_all_tm3:.6f}")
    print(f"  Significant: {'YES ✅' if p_all_tm3 < 0.05 else 'NO ❌'}\n")

    # Save results
    results = {
        "all_tms": {
            "mean": float(mean_all),
            "ci_lower": float(ci_all[0]),
            "ci_upper": float(ci_all[1]),
            "std": float(np.std(recalls_all))
        },
        "tm1_only": {
            "mean": float(mean_tm1),
            "ci_lower": float(ci_tm1[0]),
            "ci_upper": float(ci_tm1[1]),
            "std": float(np.std(recalls_tm1))
        },
        "tm3_only": {
            "mean": float(mean_tm3),
            "ci_lower": float(ci_tm3[0]),
            "ci_upper": float(ci_tm3[1]),
            "std": float(np.std(recalls_tm3))
        },
        "significance_tests": {
            "all_vs_tm1": {
                "improvement": float(mean_all - mean_tm1),
                "fold_change": float(mean_all/mean_tm1 if mean_tm1 > 0 else 0),
                "p_value": float(p_all_tm1),
                "significant": bool(p_all_tm1 < 0.05)
            },
            "all_vs_tm3": {
                "improvement": float(mean_all - mean_tm3),
                "fold_change": float(mean_all/mean_tm3 if mean_tm3 > 0 else 0),
                "p_value": float(p_all_tm3),
                "significant": bool(p_all_tm3 < 0.05)
            }
        }
    }

    with open("statistical_significance_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: statistical_significance_results.json")
    print()


if __name__ == "__main__":
    main()
