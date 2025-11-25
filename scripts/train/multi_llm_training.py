"""
Multi-LLM Training Experiment

Tests whether training on BOTH Gemini + Claude clean traces improves
cross-LLM generalization compared to single-LLM training.
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector, ModelEvaluator


def load_traces_multi_llm(gemini_dir, claude_dir, backdoor_dirs, extractor,
                          gemini_max=300, claude_max=300):
    """Load traces from both LLMs."""

    def load_from_dir(trace_dir, label, max_traces=None):
        features_list = []
        trace_files = sorted(trace_dir.glob("*.json"))

        if max_traces and len(trace_files) > max_traces:
            np.random.seed(42)
            trace_files = np.random.choice(trace_files, max_traces, replace=False)

        print(f"Loading {len(trace_files)} traces from {trace_dir.name}...")
        for trace_file in trace_files:
            try:
                with open(trace_file) as f:
                    trace = json.load(f)
                features = extractor.extract(trace)
                feature_vector = [features.get(name, 0.0) for name in extractor.get_feature_names()]
                features_list.append((feature_vector, label))
            except:
                continue

        return features_list

    # Load both LLM clean traces
    gemini_clean = load_from_dir(gemini_dir, 0, gemini_max)
    claude_clean = load_from_dir(claude_dir, 0, claude_max)

    # Load backdoors
    backdoor_traces = []
    for backdoor_dir, _ in backdoor_dirs:
        backdoor_traces.extend(load_from_dir(backdoor_dir, 1))

    return gemini_clean, claude_clean, backdoor_traces


def evaluate_model(detector, test_data, name):
    """Evaluate model on test data."""
    X_test = np.array([x[0] for x in test_data])
    y_test = np.array([x[1] for x in test_data])

    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)

    # Break down by class
    clean_mask = y_test == 0
    backdoor_mask = y_test == 1

    clean_acc = np.mean(y_pred[clean_mask] == y_test[clean_mask]) if np.any(clean_mask) else 0
    backdoor_recall = np.mean(y_pred[backdoor_mask] == y_test[backdoor_mask]) if np.any(backdoor_mask) else 0

    print(f"{name}:")
    print(f"  Overall Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Clean Accuracy: {clean_acc:.3f}")
    print(f"  Backdoor Recall: {backdoor_recall:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}\n")

    return metrics, clean_acc, backdoor_recall


def main():
    print("\n" + "="*70)
    print("MULTI-LLM TRAINING EXPERIMENT")
    print("="*70 + "\n")

    extractor = FeatureExtractor()

    # Directories
    gemini_dir = Path("data/clean_traces/gemini-2_5-flash")
    claude_dir = Path("data/clean_traces/claude-3-5-haiku@20241022")
    backdoor_dirs = [
        (Path("data/backdoor_traces/TM1"), 1),
        (Path("data/backdoor_traces/TM2"), 1),
        (Path("data/backdoor_traces/TM3"), 1),
    ]

    print("Loading traces...")
    gemini_clean, claude_clean, backdoor_traces = load_traces_multi_llm(
        gemini_dir, claude_dir, backdoor_dirs, extractor,
        gemini_max=300, claude_max=300
    )

    print(f"\nDataset Summary:")
    print(f"  Gemini clean: {len(gemini_clean)}")
    print(f"  Claude clean: {len(claude_clean)}")
    print(f"  Backdoors: {len(backdoor_traces)}")
    print()

    # Prepare test sets
    # Use 200 traces from each LLM for testing
    gemini_test = gemini_clean[:200] + backdoor_traces[:100]
    claude_test = claude_clean[:200] + backdoor_traces[100:200]

    # Experiment 1: Train on Gemini only
    print("="*70)
    print("Experiment 1: Train on Gemini Only")
    print("="*70 + "\n")

    train_gemini_only = gemini_clean[200:] + backdoor_traces[200:]
    np.random.seed(42)
    np.random.shuffle(train_gemini_only)

    X_train = np.array([x[0] for x in train_gemini_only])
    y_train = np.array([x[1] for x in train_gemini_only])

    print(f"Training data: {len(X_train)} traces ({np.sum(y_train==0)} clean, {np.sum(y_train==1)} backdoor)\n")

    detector_gemini = AnomalyDetector(model_type="svm")
    detector_gemini.fit(X_train, y_train, feature_names=extractor.get_feature_names())

    print("Test on Gemini:")
    exp1_gemini = evaluate_model(detector_gemini, gemini_test, "  Gemini")

    print("Test on Claude:")
    exp1_claude = evaluate_model(detector_gemini, claude_test, "  Claude")

    # Experiment 2: Train on Claude only
    print("="*70)
    print("Experiment 2: Train on Claude Only")
    print("="*70 + "\n")

    train_claude_only = claude_clean[200:] + backdoor_traces[200:]
    np.random.seed(42)
    np.random.shuffle(train_claude_only)

    X_train = np.array([x[0] for x in train_claude_only])
    y_train = np.array([x[1] for x in train_claude_only])

    print(f"Training data: {len(X_train)} traces ({np.sum(y_train==0)} clean, {np.sum(y_train==1)} backdoor)\n")

    detector_claude = AnomalyDetector(model_type="svm")
    detector_claude.fit(X_train, y_train, feature_names=extractor.get_feature_names())

    print("Test on Gemini:")
    exp2_gemini = evaluate_model(detector_claude, gemini_test, "  Gemini")

    print("Test on Claude:")
    exp2_claude = evaluate_model(detector_claude, claude_test, "  Claude")

    # Experiment 3: Train on BOTH Gemini + Claude
    print("="*70)
    print("Experiment 3: Train on BOTH Gemini + Claude (Multi-LLM)")
    print("="*70 + "\n")

    train_multi_llm = gemini_clean[200:] + claude_clean[200:] + backdoor_traces[200:]
    np.random.seed(42)
    np.random.shuffle(train_multi_llm)

    X_train = np.array([x[0] for x in train_multi_llm])
    y_train = np.array([x[1] for x in train_multi_llm])

    print(f"Training data: {len(X_train)} traces ({np.sum(y_train==0)} clean, {np.sum(y_train==1)} backdoor)")
    print(f"  Gemini clean: {len(gemini_clean[200:])}")
    print(f"  Claude clean: {len(claude_clean[200:])}")
    print(f"  Backdoors: {len(backdoor_traces[200:])}\n")

    detector_multi = AnomalyDetector(model_type="svm")
    detector_multi.fit(X_train, y_train, feature_names=extractor.get_feature_names())

    print("Test on Gemini:")
    exp3_gemini = evaluate_model(detector_multi, gemini_test, "  Gemini")

    print("Test on Claude:")
    exp3_claude = evaluate_model(detector_multi, claude_test, "  Claude")

    # Summary
    print("="*70)
    print("SUMMARY: Cross-LLM Generalization")
    print("="*70 + "\n")

    print("Test on Gemini:")
    print(f"  Train Gemini only → Gemini: Clean Acc={exp1_gemini[1]:.3f}, Recall={exp1_gemini[2]:.3f}")
    print(f"  Train Claude only → Gemini: Clean Acc={exp2_gemini[1]:.3f}, Recall={exp2_gemini[2]:.3f}")
    print(f"  Train Multi-LLM   → Gemini: Clean Acc={exp3_gemini[1]:.3f}, Recall={exp3_gemini[2]:.3f}")
    print()

    print("Test on Claude:")
    print(f"  Train Gemini only → Claude: Clean Acc={exp1_claude[1]:.3f}, Recall={exp1_claude[2]:.3f}")
    print(f"  Train Claude only → Claude: Clean Acc={exp2_claude[1]:.3f}, Recall={exp2_claude[2]:.3f}")
    print(f"  Train Multi-LLM   → Claude: Clean Acc={exp3_claude[1]:.3f}, Recall={exp3_claude[2]:.3f}")
    print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70 + "\n")

    gemini_improvement = exp3_gemini[1] - exp1_gemini[1]
    claude_improvement = exp3_claude[1] - exp2_claude[1]

    print(f"Multi-LLM training improves Gemini clean accuracy by: {gemini_improvement:+.3f}")
    print(f"Multi-LLM training improves Claude clean accuracy by: {claude_improvement:+.3f}")
    print()

    if gemini_improvement > 0 and claude_improvement > 0:
        print("✅ Multi-LLM training IMPROVES cross-LLM generalization")
    elif gemini_improvement > 0 or claude_improvement > 0:
        print("⚠️  Multi-LLM training PARTIALLY improves cross-LLM generalization")
    else:
        print("❌ Multi-LLM training does NOT improve cross-LLM generalization")

    print()

    # Save results
    results = {
        "experiment_1_gemini_only": {
            "test_gemini": {"clean_acc": float(exp1_gemini[1]), "recall": float(exp1_gemini[2])},
            "test_claude": {"clean_acc": float(exp1_claude[1]), "recall": float(exp1_claude[2])}
        },
        "experiment_2_claude_only": {
            "test_gemini": {"clean_acc": float(exp2_gemini[1]), "recall": float(exp2_gemini[2])},
            "test_claude": {"clean_acc": float(exp2_claude[1]), "recall": float(exp2_claude[2])}
        },
        "experiment_3_multi_llm": {
            "test_gemini": {"clean_acc": float(exp3_gemini[1]), "recall": float(exp3_gemini[2])},
            "test_claude": {"clean_acc": float(exp3_claude[1]), "recall": float(exp3_claude[2])}
        }
    }

    with open("multi_llm_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("Results saved to: multi_llm_training_results.json")
    print()


if __name__ == "__main__":
    main()
