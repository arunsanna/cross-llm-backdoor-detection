"""
Data-flow feature extraction.

Captures data movement and dependencies:
- Input-output relationships
- Data size patterns
- Information flow anomalies
- Sensitive data handling patterns
"""

from typing import Dict, Any, List
import numpy as np
import re
from .base import BaseFeatureExtractor


class DataFlowFeatureExtractor(BaseFeatureExtractor):
    """Extract data-flow behavioral features."""

    def __init__(self):
        super().__init__("dataflow_level")
        # Patterns that might indicate sensitive data or exfiltration
        self.sensitive_patterns = [
            r"password", r"secret", r"token", r"api[_-]?key",
            r"credential", r"auth", r"private", r"confidential"
        ]

    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """Extract data-flow features from a trace."""
        steps = trace.get("steps", [])

        if not steps:
            return self._empty_features()

        features = {}

        input_sizes = []
        output_sizes = []
        data_dependencies = []

        for i, step in enumerate(steps):
            # Input/output size analysis (use data_in and data_out from actual traces)
            input_data = step.get("data_in", {})
            output_data = step.get("data_out", {})

            input_size = len(str(input_data))
            output_size = len(str(output_data))

            input_sizes.append(input_size)
            output_sizes.append(output_size)

            # Check for data dependencies (output of step i used in step i+1)
            if i < len(steps) - 1:
                next_input = steps[i + 1].get("data_in", {})
                if output_data and str(output_data) in str(next_input):
                    data_dependencies.append(1)
                else:
                    data_dependencies.append(0)

        # Feature 1-4: Input data characteristics
        features["avg_input_size"] = np.mean(input_sizes) if input_sizes else 0
        features["max_input_size"] = max(input_sizes) if input_sizes else 0
        features["std_input_size"] = np.std(input_sizes) if input_sizes else 0
        features["input_size_variation"] = features["std_input_size"] / features["avg_input_size"] if features["avg_input_size"] > 0 else 0

        # Feature 5-8: Output data characteristics
        features["avg_output_size"] = np.mean(output_sizes) if output_sizes else 0
        features["max_output_size"] = max(output_sizes) if output_sizes else 0
        features["std_output_size"] = np.std(output_sizes) if output_sizes else 0
        features["output_size_variation"] = features["std_output_size"] / features["avg_output_size"] if features["avg_output_size"] > 0 else 0

        # Feature 9-10: Data expansion/compression ratios
        ratios = [out / inp if inp > 0 else 0 for inp, out in zip(input_sizes, output_sizes)]
        features["avg_io_ratio"] = np.mean(ratios) if ratios else 0
        features["max_io_ratio"] = max(ratios) if ratios else 0

        # Feature 11-12: Data dependency patterns
        features["dependency_ratio"] = np.mean(data_dependencies) if data_dependencies else 0
        features["total_dependencies"] = sum(data_dependencies)

        # Feature 13-14: Sensitive data patterns (backdoor indicator)
        features["sensitive_data_mentions"] = self._count_sensitive_patterns(steps)
        features["has_sensitive_data"] = float(features["sensitive_data_mentions"] > 0)

        # Feature 15: Data flow complexity
        features["data_flow_complexity"] = self._calculate_flow_complexity(steps)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            "avg_input_size",
            "max_input_size",
            "std_input_size",
            "input_size_variation",
            "avg_output_size",
            "max_output_size",
            "std_output_size",
            "output_size_variation",
            "avg_io_ratio",
            "max_io_ratio",
            "dependency_ratio",
            "total_dependencies",
            "sensitive_data_mentions",
            "has_sensitive_data",
            "data_flow_complexity",
        ]

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features for traces with no steps."""
        return {name: 0.0 for name in self.get_feature_names()}

    def _count_sensitive_patterns(self, steps: List[Dict[str, Any]]) -> int:
        """Count mentions of sensitive data patterns."""
        count = 0

        for step in steps:
            # Check data_in, data_out, and params (actual trace format)
            text = " ".join([
                str(step.get("data_in", {})),
                str(step.get("data_out", {})),
                str(step.get("params", {}))
            ]).lower()

            for pattern in self.sensitive_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    count += 1

        return count

    def _calculate_flow_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """
        Calculate complexity of data flow.
        Based on number of unique data transformations and dependencies.
        """
        if not steps:
            return 0.0

        # Count unique tool types (different transformation types)
        # Handle tool as dict or string
        tools = []
        for step in steps:
            tool = step.get("tool", "unknown")
            if isinstance(tool, dict):
                tool = tool.get("name", "unknown")
            tools.append(tool)
        unique_tools = len(set(tools))

        # Count steps with multiple inputs (complex dependencies)
        complex_steps = 0
        for step in steps:
            params = step.get("params", {})
            if isinstance(params, dict) and len(params) > 2:
                complex_steps += 1

        # Normalize by total steps
        complexity = (unique_tools + complex_steps) / len(steps)
        return complexity
