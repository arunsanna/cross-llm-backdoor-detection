"""
Main feature extractor that combines all 4 dimensions.
"""

from typing import Dict, Any, List
import numpy as np
from pathlib import Path
import json

from .action_features import ActionFeatureExtractor
from .sequence_features import SequenceFeatureExtractor
from .dataflow_features import DataFlowFeatureExtractor
from .temporal_features import TemporalFeatureExtractor


class FeatureExtractor:
    """
    Main feature extractor that combines all 4 dimensions:
    1. Action-level
    2. Sequence-level
    3. Data-flow
    4. Temporal
    """

    def __init__(self):
        self.extractors = [
            ActionFeatureExtractor(),
            SequenceFeatureExtractor(),
            DataFlowFeatureExtractor(),
            TemporalFeatureExtractor(),
        ]

    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract all features from a trace.

        Args:
            trace: Agent execution trace

        Returns:
            Dictionary of feature_name -> feature_value
        """
        features = {}

        for extractor in self.extractors:
            extractor_features = extractor.extract(trace)
            features.update(extractor_features)

        return features

    def extract_from_file(self, trace_file: Path) -> Dict[str, float]:
        """
        Extract features from a trace file.

        Args:
            trace_file: Path to JSON trace file

        Returns:
            Dictionary of features
        """
        with open(trace_file, 'r') as f:
            trace = json.load(f)

        return self.extract(trace)

    def extract_batch(self, traces: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract features from multiple traces.

        Args:
            traces: List of agent execution traces

        Returns:
            2D numpy array of shape (n_traces, n_features)
        """
        feature_dicts = [self.extract(trace) for trace in traces]
        feature_names = self.get_feature_names()

        # Convert to numpy array
        features = []
        for feat_dict in feature_dicts:
            features.append([feat_dict.get(name, 0.0) for name in feature_names])

        return np.array(features)

    def extract_from_directory(self, trace_dir: Path) -> np.ndarray:
        """
        Extract features from all traces in a directory.

        Args:
            trace_dir: Directory containing trace JSON files

        Returns:
            2D numpy array of features
        """
        trace_files = sorted(trace_dir.glob("*.json"))

        if not trace_files:
            raise ValueError(f"No trace files found in {trace_dir}")

        print(f"Extracting features from {len(trace_files)} traces in {trace_dir}")

        traces = []
        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace = json.load(f)
                    traces.append(trace)
            except Exception as e:
                print(f"Error loading {trace_file}: {e}")
                continue

        return self.extract_batch(traces)

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names across all extractors.

        Returns:
            List of feature names
        """
        feature_names = []
        for extractor in self.extractors:
            feature_names.extend(extractor.get_feature_names())
        return feature_names

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_feature_names())

    def get_feature_info(self) -> Dict[str, List[str]]:
        """
        Get feature information grouped by extractor.

        Returns:
            Dictionary mapping extractor name to list of features
        """
        info = {}
        for extractor in self.extractors:
            info[extractor.name] = extractor.get_feature_names()
        return info

    def print_feature_summary(self):
        """Print summary of available features."""
        print("=" * 70)
        print("FEATURE EXTRACTION SUMMARY")
        print("=" * 70)
        print()

        info = self.get_feature_info()

        for extractor_name, features in info.items():
            print(f"{extractor_name}:")
            print(f"  {len(features)} features")
            for i, feat in enumerate(features, 1):
                print(f"    {i}. {feat}")
            print()

        print(f"Total features: {self.get_feature_count()}")
        print("=" * 70)
