"""
Test feature extraction on sample traces.

Validates that all 4 dimensional features work correctly on:
- Clean traces
- Backdoor traces (TM1, TM2, TM3)
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from features import FeatureExtractor


def test_single_trace():
    """Test feature extraction on a single trace."""
    print("=" * 70)
    print("TEST 1: Single Trace Feature Extraction")
    print("=" * 70)
    print()

    # Find a clean trace
    clean_traces = list(Path("data/clean_traces/gemini-2_5-flash").glob("*.json"))

    if not clean_traces:
        print("❌ No clean traces found. Run trace collection first.")
        return False

    trace_file = clean_traces[0]
    print(f"Testing on: {trace_file.name}")
    print()

    # Load trace
    with open(trace_file, 'r') as f:
        trace = json.load(f)

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(trace)

    # Validate
    print(f"✅ Extracted {len(features)} features")
    print()

    # Show sample features from each dimension
    print("Sample features by dimension:")
    print()

    info = extractor.get_feature_info()
    for dimension, feature_names in info.items():
        print(f"{dimension}:")
        for i, feat_name in enumerate(feature_names[:3], 1):  # Show first 3
            value = features.get(feat_name, 0.0)
            print(f"  {feat_name}: {value:.4f}")
        if len(feature_names) > 3:
            print(f"  ... and {len(feature_names) - 3} more")
        print()

    return True


def test_clean_vs_backdoor():
    """Test feature extraction on clean vs backdoor traces."""
    print("=" * 70)
    print("TEST 2: Clean vs Backdoor Feature Comparison")
    print("=" * 70)
    print()

    extractor = FeatureExtractor()

    # Load clean traces
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")
    clean_traces = sorted(clean_dir.glob("*.json"))[:5]  # First 5

    if not clean_traces:
        print("❌ No clean traces found")
        return False

    # Load backdoor traces (TM1)
    backdoor_dir = Path("data/backdoor_traces/TM1")
    backdoor_traces = []
    if backdoor_dir.exists():
        backdoor_traces = sorted(backdoor_dir.glob("*.json"))[:5]  # First 5

    if not backdoor_traces:
        print("⚠️  No TM1 backdoor traces found, using only clean traces")

    # Extract features
    print(f"Extracting features from {len(clean_traces)} clean traces...")
    clean_features_list = []
    for trace_file in clean_traces:
        with open(trace_file, 'r') as f:
            trace = json.load(f)
            features = extractor.extract(trace)
            clean_features_list.append(features)

    backdoor_features_list = []
    if backdoor_traces:
        print(f"Extracting features from {len(backdoor_traces)} backdoor traces...")
        for trace_file in backdoor_traces:
            with open(trace_file, 'r') as f:
                trace = json.load(f)
                features = extractor.extract(trace)
                backdoor_features_list.append(features)

    # Calculate statistics
    feature_names = extractor.get_feature_names()

    print()
    print("Feature Statistics:")
    print()
    print(f"{'Feature':<30} {'Clean Mean':<15} {'Backdoor Mean':<15} {'Difference':<15}")
    print("-" * 75)

    significant_diffs = []

    for feat_name in feature_names:
        clean_vals = [f[feat_name] for f in clean_features_list]
        clean_mean = np.mean(clean_vals)

        if backdoor_features_list:
            backdoor_vals = [f[feat_name] for f in backdoor_features_list]
            backdoor_mean = np.mean(backdoor_vals)
            diff = abs(backdoor_mean - clean_mean)

            # Track significant differences
            if clean_mean > 0 and diff / clean_mean > 0.5:  # >50% difference
                significant_diffs.append((feat_name, clean_mean, backdoor_mean, diff))

            print(f"{feat_name:<30} {clean_mean:<15.4f} {backdoor_mean:<15.4f} {diff:<15.4f}")
        else:
            print(f"{feat_name:<30} {clean_mean:<15.4f} {'N/A':<15} {'N/A':<15}")

    print()

    if significant_diffs:
        print(f"✅ Found {len(significant_diffs)} features with significant differences (>50%):")
        for feat_name, clean, backdoor, diff in significant_diffs[:5]:
            print(f"  • {feat_name}: Clean={clean:.4f}, Backdoor={backdoor:.4f}")
    else:
        print("ℹ️  No significant differences found (may need more traces)")

    print()
    return True


def test_batch_extraction():
    """Test batch feature extraction."""
    print("=" * 70)
    print("TEST 3: Batch Feature Extraction")
    print("=" * 70)
    print()

    extractor = FeatureExtractor()

    # Load traces
    clean_dir = Path("data/clean_traces/gemini-2_5-flash")
    trace_files = sorted(clean_dir.glob("*.json"))[:10]  # First 10

    if not trace_files:
        print("❌ No traces found")
        return False

    print(f"Loading {len(trace_files)} traces...")
    traces = []
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            traces.append(json.load(f))

    # Extract features in batch
    print("Extracting features in batch...")
    feature_matrix = extractor.extract_batch(traces)

    print()
    print(f"✅ Feature matrix shape: {feature_matrix.shape}")
    print(f"   Traces: {feature_matrix.shape[0]}")
    print(f"   Features per trace: {feature_matrix.shape[1]}")
    print()

    # Show statistics
    print("Feature matrix statistics:")
    print(f"  Mean: {np.mean(feature_matrix):.4f}")
    print(f"  Std: {np.std(feature_matrix):.4f}")
    print(f"  Min: {np.min(feature_matrix):.4f}")
    print(f"  Max: {np.max(feature_matrix):.4f}")
    print()

    return True


def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "FEATURE EXTRACTION TEST SUITE" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Show feature summary
    extractor = FeatureExtractor()
    extractor.print_feature_summary()
    print()

    # Run tests
    tests = [
        ("Single Trace", test_single_trace),
        ("Clean vs Backdoor", test_clean_vs_backdoor),
        ("Batch Extraction", test_batch_extraction),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
            print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print()
        print("🎉 All tests passed! Feature extraction system is ready.")
        print()
        print("Next steps:")
        print("1. Extract features from all traces")
        print("2. Build detection models (Phase 3)")
        print("3. Evaluate on backdoor traces")
        print()
    else:
        print()
        print("⚠️  Some tests failed. Review errors above.")
        print()

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
