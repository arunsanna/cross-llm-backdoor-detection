"""
RQ4: Detection Overhead Measurement

Measures the computational cost of backdoor detection:
1. Feature extraction latency
2. Model inference time
3. Memory usage
4. End-to-end detection time
"""

import sys
from pathlib import Path
import json
import numpy as np
import time
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor
from models import AnomalyDetector


def measure_feature_extraction(extractor, trace_files, n_samples=100):
    """Measure feature extraction latency."""
    print("Measuring feature extraction latency...")

    np.random.seed(42)
    sample_files = np.random.choice(trace_files, min(n_samples, len(trace_files)), replace=False)

    times = []
    for trace_file in sample_files:
        with open(trace_file) as f:
            trace = json.load(f)

        start = time.perf_counter()
        features = extractor.extract(trace)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def measure_inference(detector, X_test, n_runs=100):
    """Measure model inference latency."""
    print(f"Measuring inference latency ({n_runs} runs)...")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = detector.predict(X_test[:1])  # Single prediction
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def measure_memory(detector):
    """Measure model memory usage."""
    print("Measuring memory usage...")

    import pickle
    import tempfile

    # Save model to measure size
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(detector, f)
        model_size_kb = os.path.getsize(f.name) / 1024
        os.unlink(f.name)

    # Measure process memory
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "model_size_kb": model_size_kb,
        "model_size_mb": model_size_kb / 1024,
        "process_memory_mb": memory_info.rss / (1024 * 1024),
    }


def measure_end_to_end(extractor, detector, trace_files, n_samples=50):
    """Measure end-to-end detection time (feature extraction + inference)."""
    print(f"Measuring end-to-end latency ({n_samples} traces)...")

    np.random.seed(42)
    sample_files = np.random.choice(trace_files, min(n_samples, len(trace_files)), replace=False)

    times = []
    for trace_file in sample_files:
        with open(trace_file) as f:
            trace = json.load(f)

        start = time.perf_counter()
        # Feature extraction
        features = extractor.extract(trace)
        feature_vector = np.array([[features.get(name, 0.0) for name in extractor.get_feature_names()]])
        # Inference
        _ = detector.predict(feature_vector)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    print("\n" + "="*70)
    print("RQ4: DETECTION OVERHEAD MEASUREMENT")
    print("="*70 + "\n")

    # Load test data
    print("Loading test data...")
    extractor = FeatureExtractor()

    clean_dir = Path("data/clean_traces/gemini-2_5-flash")
    trace_files = [f for f in clean_dir.glob("*.json") if "collection_errors" not in f.name][:200]

    print(f"Loaded {len(trace_files)} test traces\n")

    # Extract features for one batch (for inference measurement)
    print("Extracting features for test set...")
    X_test = []
    for trace_file in trace_files[:50]:
        try:
            with open(trace_file) as f:
                trace = json.load(f)
            features = extractor.extract(trace)
            feature_vector = [features.get(name, 0.0) for name in extractor.get_feature_names()]
            X_test.append(feature_vector)
        except:
            continue

    X_test = np.array(X_test)
    print(f"Test set: {len(X_test)} traces\n")

    # Load trained models
    print("Loading trained models...\n")
    rf_detector = AnomalyDetector.load(Path("models/random_forest_detector.pkl"))
    svm_detector = AnomalyDetector.load(Path("models/svm_detector.pkl"))

    results = {}

    # Measure overhead for each model
    for model_name, detector in [("Random Forest", rf_detector), ("SVM", svm_detector)]:
        print("="*70)
        print(f"{model_name} Overhead")
        print("="*70 + "\n")

        model_results = {}

        # 1. Feature extraction (model-independent)
        if "feature_extraction" not in results:
            results["feature_extraction"] = measure_feature_extraction(extractor, trace_files, n_samples=100)

        # 2. Inference time
        model_results["inference"] = measure_inference(detector, X_test, n_runs=100)

        # 3. Memory usage
        model_results["memory"] = measure_memory(detector)

        # 4. End-to-end
        model_results["end_to_end"] = measure_end_to_end(extractor, detector, trace_files, n_samples=50)

        results[model_name.lower().replace(" ", "_")] = model_results

        print()

    # Print summary
    print("="*70)
    print("OVERHEAD SUMMARY")
    print("="*70 + "\n")

    print("Feature Extraction (model-independent):")
    print(f"  Mean:   {results['feature_extraction']['mean_ms']:.2f} ms")
    print(f"  Median: {results['feature_extraction']['median_ms']:.2f} ms")
    print(f"  P95:    {results['feature_extraction']['p95_ms']:.2f} ms")
    print()

    for model_key, model_name in [("random_forest", "Random Forest"), ("svm", "SVM")]:
        print(f"{model_name}:")
        print(f"  Inference Mean:   {results[model_key]['inference']['mean_ms']:.3f} ms")
        print(f"  Inference Median: {results[model_key]['inference']['median_ms']:.3f} ms")
        print(f"  End-to-End Mean:  {results[model_key]['end_to_end']['mean_ms']:.2f} ms")
        print(f"  End-to-End P95:   {results[model_key]['end_to_end']['p95_ms']:.2f} ms")
        print(f"  Model Size:       {results[model_key]['memory']['model_size_mb']:.2f} MB")
        print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70 + "\n")

    feat_mean = results['feature_extraction']['mean_ms']
    rf_inf_mean = results['random_forest']['inference']['mean_ms']
    svm_inf_mean = results['svm']['inference']['mean_ms']

    print(f"Feature extraction accounts for {feat_mean/(feat_mean+rf_inf_mean)*100:.1f}% of RF detection time")
    print(f"Feature extraction accounts for {feat_mean/(feat_mean+svm_inf_mean)*100:.1f}% of SVM detection time")
    print()

    print(f"Random Forest is {svm_inf_mean/rf_inf_mean:.1f}x faster than SVM for inference")
    print()

    rf_throughput = 1000 / results['random_forest']['end_to_end']['mean_ms']
    svm_throughput = 1000 / results['svm']['end_to_end']['mean_ms']
    print(f"Random Forest throughput: {rf_throughput:.0f} traces/second")
    print(f"SVM throughput:           {svm_throughput:.0f} traces/second")
    print()

    # Save results
    output_file = Path("overhead_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
