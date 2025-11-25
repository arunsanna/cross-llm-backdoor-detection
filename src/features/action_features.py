"""
Action-level feature extraction.

Captures characteristics of individual tool calls and actions:
- Tool usage frequency and diversity
- Parameter patterns and anomalies
- Action success rates
- Unusual tool combinations
"""

from typing import Dict, Any, List
from collections import Counter
import numpy as np
from .base import BaseFeatureExtractor


class ActionFeatureExtractor(BaseFeatureExtractor):
    """Extract action-level behavioral features."""

    def __init__(self):
        super().__init__("action_level")

    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """Extract action-level features from a trace."""
        steps = trace.get("steps", [])

        if not steps:
            return self._empty_features()

        features = {}

        # Tool usage statistics
        tool_counts = Counter()
        param_lengths = []

        for step in steps:
            # Handle tool as dict or string
            tool = step.get("tool", "unknown")
            if isinstance(tool, dict):
                tool = tool.get("name", "unknown")
            tool_counts[tool] += 1

            # Parameter characteristics (params field in actual traces)
            params = step.get("params", {})
            if isinstance(params, dict):
                param_lengths.append(len(str(params)))

        # Feature 1-3: Tool diversity metrics
        features["tool_diversity"] = len(tool_counts)  # Number of unique tools
        features["tool_entropy"] = self._calculate_entropy(tool_counts)
        features["max_tool_freq"] = max(tool_counts.values()) if tool_counts else 0

        # Feature 4-6: Tool usage patterns
        features["total_actions"] = len(steps)
        features["avg_params_length"] = np.mean(param_lengths) if param_lengths else 0
        features["max_params_length"] = max(param_lengths) if param_lengths else 0

        # Feature 7-9: Rare tool usage (potential backdoor indicators)
        common_tools = {"Search", "Calculator", "Summarize"}
        rare_tool_count = sum(1 for tool in tool_counts if tool not in common_tools)
        features["rare_tool_ratio"] = rare_tool_count / len(tool_counts) if tool_counts else 0
        features["rare_tool_count"] = rare_tool_count

        # Feature 10: Tool usage concentration (Gini coefficient)
        features["tool_concentration"] = self._gini_coefficient(list(tool_counts.values()))

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            "tool_diversity",
            "tool_entropy",
            "max_tool_freq",
            "total_actions",
            "avg_params_length",
            "max_params_length",
            "rare_tool_ratio",
            "rare_tool_count",
            "tool_concentration",
        ]

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features for traces with no steps."""
        return {name: 0.0 for name in self.get_feature_names()}

    def _calculate_entropy(self, counts: Counter) -> float:
        """Calculate Shannon entropy of tool usage."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        probs = [count / total for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def _gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient (measure of inequality)."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)

        return (2 * sum((i + 1) * val for i, val in enumerate(sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
