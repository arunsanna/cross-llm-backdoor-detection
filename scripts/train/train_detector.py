"""
Train and evaluate backdoor detection models.

Usage:
    python train_detector.py --model random_forest
    python train_detector.py --model svm --threat-model TM1
    python train_detector.py --model all  # Train all models
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector, ModelEvaluator


def load_and_extract_features(trace_dirs, extractor, max_clean_traces=None):
    """
    Load traces and extract features.

    Args:
        trace_dirs: List of (directory, label) tuples
        extractor: FeatureExtractor instance
        max_clean_traces: Maximum number of clean traces to load (None = all)

    Returns:
        Features (X) and labels (y)
    """
    all_features = []
    all_labels = []

    for trace_dir, label in trace_dirs:
        if not trace_dir.exists():
            print(f"⚠️  Directory not found: {trace_dir}")
            continue

        trace_files = sorted(trace_dir.glob("*.json"))

        if not trace_files:
            print(f"⚠️  No traces found in: {trace_dir}")
            continue

        # Subsample clean traces if requested
        if label == 0 and max_clean_traces is not None and len(trace_files) > max_clean_traces:
            np.random.seed(42)
            trace_files = np.random.choice(trace_files, max_clean_traces, replace=False)
            print(f"Subsampling {max_clean_traces} clean traces from {trace_dir.name} (had {len(sorted(trace_dir.glob('*.json')))} total)...")
        else:
            print(f"Loading {len(trace_files)} traces from {trace_dir.name}...")

        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace = json.load(f)

                features = extractor.extract(trace)
                feature_vector = [features.get(name, 0.0) for name in extractor.get_feature_names()]

                all_features.append(feature_vector)
                all_labels.append(label)

            except Exception as e:
                print(f"Error loading {trace_file.name}: {e}")
                continue

    X = np.array(all_features)
    y = np.array(all_labels)

    return X, y


def train_single_model(model_type, X_train, y_train, X_test, y_test, feature_names, output_dir):
    """Train and evaluate a single model."""
    print()
    print("=" * 70)
    print(f"Training {model_type.upper()} Model")
    print("=" * 70)
    print()

    # Create detector
    detector = AnomalyDetector(model_type=model_type)

    # Train
    print(f"Training on {len(X_train)} samples...")
    detector.fit(X_train, y_train, feature_names=feature_names)

    # Evaluate
    print(f"Evaluating on {len(X_test)} samples...")
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)

    # Compute metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)

    # Print results
    evaluator.print_metrics(metrics, title=f"{model_type.upper()} Performance")

    # Feature importance (if available)
    if model_type == "random_forest":
        print("Top 10 Most Important Features:")
        importance = detector.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feat_name, imp) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {feat_name:<30} {imp:.6f}")
        print()

    # Save model
    model_path = output_dir / f"{model_type}_detector.pkl"
    detector.save(model_path)
    print(f"Model saved to: {model_path}")
    print()

    return metrics, detector


def main():
    parser = argparse.ArgumentParser(description="Train backdoor detection models")

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm", "isolation_forest", "all"],
        help="Model type to train"
    )

    parser.add_argument(
        "--threat-model",
        type=str,
        default=None,
        choices=["TM1", "TM2", "TM3", "all"],
        help="Specific threat model to evaluate (None = all backdoors)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (default: 0.2)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models"
    )

    parser.add_argument(
        "--max-clean-traces",
        type=int,
        default=None,
        help="Maximum number of clean traces to use (for balancing dataset)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "BACKDOOR DETECTION TRAINING" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Initialize feature extractor
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()

    print(f"Feature extractor ready: {len(feature_names)} features")
    print()

    # Load traces
    print("=" * 70)
    print("LOADING TRACES")
    print("=" * 70)
    print()

    # Clean traces
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")

    # Backdoor traces
    if args.threat_model and args.threat_model != "all":
        backdoor_dirs = [(Path(f"data/backdoor_traces/{args.threat_model}"), 1)]
        print(f"Evaluating against: {args.threat_model}")
    else:
        backdoor_dirs = [
            (Path("data/backdoor_traces/TM1"), 1),
            (Path("data/backdoor_traces/TM2"), 1),
            (Path("data/backdoor_traces/TM3"), 1),
        ]
        print("Evaluating against: All threat models (TM1, TM2, TM3)")

    print()

    # Load clean traces
    trace_dirs = [(clean_dir, 0)]  # label=0 for clean
    trace_dirs.extend(backdoor_dirs)

    X, y = load_and_extract_features(trace_dirs, extractor, max_clean_traces=args.max_clean_traces)

    if len(X) == 0:
        print()
        print("❌ No traces loaded. Run trace collection first.")
        return 1

    print()
    print(f"Dataset Summary:")
    print(f"  Total traces: {len(X)}")
    print(f"  Clean traces: {np.sum(y == 0)}")
    print(f"  Backdoor traces: {np.sum(y == 1)}")
    print(f"  Features per trace: {X.shape[1]}")
    print()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"Train/Test Split:")
    print(f"  Train: {len(X_train)} samples ({np.sum(y_train == 0)} clean, {np.sum(y_train == 1)} backdoor)")
    print(f"  Test:  {len(X_test)} samples ({np.sum(y_test == 0)} clean, {np.sum(y_test == 1)} backdoor)")
    print()

    # Train models
    if args.model == "all":
        models_to_train = ["random_forest", "svm", "isolation_forest"]
    else:
        models_to_train = [args.model]

    all_results = {}

    for model_type in models_to_train:
        try:
            metrics, detector = train_single_model(
                model_type, X_train, y_train, X_test, y_test,
                feature_names, output_dir
            )
            all_results[model_type] = metrics
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compare models if multiple trained
    if len(all_results) > 1:
        evaluator = ModelEvaluator()
        evaluator.compare_models(all_results)

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Models saved to: {output_dir}/")
    print()
    print("Next steps:")
    print("1. Evaluate on specific threat models: --threat-model TM1")
    print("2. Test on new traces")
    print("3. Analyze feature importance")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
