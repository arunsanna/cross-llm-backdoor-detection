"""
Quick test of trace collection pipeline

Tests:
1. Agent initialization with Vertex AI
2. Task execution
3. Trace saving with model metadata
4. Collection statistics
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from collect_traces import collect_traces

def test_small_collection():
    """Test collection of 5 traces"""
    print("Testing trace collection pipeline with 5 traces...\n")

    stats = collect_traces(
        model_name="gemini-2.5-flash",
        num_traces=5,
        batch_size=1,
        output_dir=Path("data/clean_traces/test_collection"),
        retry_attempts=2
    )

    # Verify results
    assert stats.total_successful >= 3, f"Expected at least 3 successful traces, got {stats.total_successful}"

    print("\n✅ Test passed!")
    print(f"Collected {stats.total_successful}/5 traces successfully")

    return stats


if __name__ == "__main__":
    test_small_collection()
