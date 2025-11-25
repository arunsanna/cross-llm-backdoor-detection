"""
RQ6: Cross-LLM Generalization Experiment

Evaluates whether backdoor detectors trained on one LLM (Gemini) can detect
backdoors in traces from another LLM (Claude) and vice versa.

Experiments:
1. Train on Gemini clean + all backdoors → Test on Claude clean
2. Train on Claude clean + all backdoors → Test on Gemini clean
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector, ModelEvaluator


def load_traces_by_llm(gemini_dir, claude_dir, backdoor_dirs, extractor, max_clean=600):
    """Load traces separated by LLM."""

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

    # Load Gemini clean
    gemini_clean = load_from_dir(gemini_dir, 0, max_clean)

    # Load Claude clean
    claude_clean = load_from_dir(claude_dir, 0, max_clean)

    # Load backdoors (same for both experiments)
    backdoor_traces = []
    for backdoor_dir, _ in backdoor_dirs:
        backdoor_traces.extend(load_from_dir(backdoor_dir, 1))

    return gemini_clean, claude_clean, backdoor_traces


def evaluate_cross_llm(train_clean, test_clean, backdoor_traces, train_name, test_name, extractor):
    """Train on one LLM, test on another."""

    print("\n" + "="*70)
    print(f"Experiment: Train on {train_name} → Test on {test_name}")
    print("="*70 + "\n")

    # Prepare training data (train_clean + all backdoors)
    train_data = train_clean + backdoor_traces
    np.random.seed(42)
    np.random.shuffle(train_data)

    X_train = np.array([x[0] for x in train_data])
    y_train = np.array([x[1] for x in train_data])

    # Test data (test_clean only - we're testing if it can correctly identify clean traces from different LLM)
    X_test_clean = np.array([x[0] for x in test_clean])
    y_test_clean = np.array([0] * len(test_clean))

    # Also test on backdoors to measure recall
    np.random.seed(42)
    backdoor_test = np.random.choice(len(backdoor_traces), min(200, len(backdoor_traces)), replace=False)
    X_test_backdoor = np.array([backdoor_traces[i][0] for i in backdoor_test])
    y_test_backdoor = np.array([1] * len(backdoor_test))

    # Combined test set
    X_test = np.vstack([X_test_clean, X_test_backdoor])
    y_test = np.concatenate([y_test_clean, y_test_backdoor])

    print(f"Training Data:")
    print(f"  {train_name} clean: {np.sum(y_train == 0)}")
    print(f"  Backdoors: {np.sum(y_train == 1)}")
    print(f"  Total: {len(X_train)}\n")

    print(f"Test Data:")
    print(f"  {test_name} clean: {len(X_test_clean)}")
    print(f"  Backdoors: {len(X_test_backdoor)}")
    print(f"  Total: {len(X_test)}\n")

    # Train models
    results = {}
    evaluator = ModelEvaluator()

    for model_type in ['random_forest', 'svm']:
        print(f"Training {model_type.upper()}...")
        detector = AnomalyDetector(model_type=model_type)
        detector.fit(X_train, y_train, feature_names=extractor.get_feature_names())

        y_pred = detector.predict(X_test)
        y_proba = detector.predict_proba(X_test)

        metrics = evaluator.evaluate(y_test, y_pred, y_proba)
        results[model_type] = metrics

        # Break down by test set type
        clean_pred = y_pred[:len(X_test_clean)]
        clean_accuracy = np.mean(clean_pred == y_test_clean)

        backdoor_pred = y_pred[len(X_test_clean):]
        backdoor_recall = np.mean(backdoor_pred == y_test_backdoor)

        print(f"  Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"  {test_name} Clean Accuracy: {clean_accuracy:.3f}")
        print(f"  Backdoor Recall: {backdoor_recall:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f}\n")

    return results


def main():
    print("\n" + "="*70)
    print("RQ6: CROSS-LLM GENERALIZATION EXPERIMENT")
    print("="*70 + "\n")

    start_time = time.time()

    # Initialize
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
    gemini_clean, claude_clean, backdoor_traces = load_traces_by_llm(
        gemini_dir, claude_dir, backdoor_dirs, extractor, max_clean=600
    )

    print(f"\nDataset Summary:")
    print(f"  Gemini clean: {len(gemini_clean)}")
    print(f"  Claude clean: {len(claude_clean)}")
    print(f"  Backdoors: {len(backdoor_traces)}")
    print()

    # Experiment 1: Train on Gemini → Test on Claude
    exp1_results = evaluate_cross_llm(
        gemini_clean, claude_clean, backdoor_traces,
        "Gemini", "Claude", extractor
    )

    # Experiment 2: Train on Claude → Test on Gemini
    exp2_results = evaluate_cross_llm(
        claude_clean, gemini_clean, backdoor_traces,
        "Claude", "Gemini", extractor
    )

    # Summary
    print("\n" + "="*70)
    print("CROSS-LLM GENERALIZATION SUMMARY")
    print("="*70 + "\n")

    print("Experiment 1: Train on Gemini → Test on Claude")
    print("-"*70)
    for model_type, metrics in exp1_results.items():
        print(f"{model_type.upper():<20} Accuracy: {metrics['accuracy']:.3f}  "
              f"Recall: {metrics['recall']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}")
    print()

    print("Experiment 2: Train on Claude → Test on Gemini")
    print("-"*70)
    for model_type, metrics in exp2_results.items():
        print(f"{model_type.upper():<20} Accuracy: {metrics['accuracy']:.3f}  "
              f"Recall: {metrics['recall']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}")
    print()

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")
    print()

    # Save results
    output_file = Path("cross_llm_results.json")
    results_data = {
        "gemini_to_claude": {k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float))}
                             for k, v in exp1_results.items()},
        "claude_to_gemini": {k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, (int, float))}
                            for k, v in exp2_results.items()},
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
