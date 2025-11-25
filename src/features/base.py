"""
Base feature extractor interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from a trace.

        Args:
            trace: Agent execution trace

        Returns:
            Dictionary of feature_name -> feature_value
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names produced by this extractor.

        Returns:
            List of feature names
        """
        pass

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
