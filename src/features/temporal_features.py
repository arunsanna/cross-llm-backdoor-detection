"""
Temporal feature extraction.

Captures timing patterns and anomalies:
- Action duration distributions
- Inter-action delays
- Execution speed anomalies
- Time-based attack indicators
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from .base import BaseFeatureExtractor


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal behavioral features."""

    def __init__(self):
        super().__init__("temporal_level")

    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """Extract temporal features from a trace."""
        steps = trace.get("steps", [])

        if not steps:
            return self._empty_features()

        features = {}

        # Extract timing information
        durations = []
        inter_action_delays = []
        timestamps = []

        for step in steps:
            duration = step.get("duration_ms", 0)
            durations.append(duration)

            timestamp = step.get("timestamp", None)
            if timestamp:
                timestamps.append(timestamp)

        # Calculate inter-action delays
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                try:
                    t1 = datetime.fromisoformat(timestamps[i - 1].replace('Z', '+00:00'))
                    t2 = datetime.fromisoformat(timestamps[i].replace('Z', '+00:00'))
                    delay = (t2 - t1).total_seconds() * 1000  # Convert to ms
                    inter_action_delays.append(delay)
                except:
                    pass

        # Feature 1-4: Action duration statistics
        features["avg_duration"] = np.mean(durations) if durations else 0
        features["max_duration"] = max(durations) if durations else 0
        features["std_duration"] = np.std(durations) if durations else 0
        features["duration_variation"] = features["std_duration"] / features["avg_duration"] if features["avg_duration"] > 0 else 0

        # Feature 5-8: Inter-action delay statistics
        features["avg_delay"] = np.mean(inter_action_delays) if inter_action_delays else 0
        features["max_delay"] = max(inter_action_delays) if inter_action_delays else 0
        features["std_delay"] = np.std(inter_action_delays) if inter_action_delays else 0
        features["delay_variation"] = features["std_delay"] / features["avg_delay"] if features["avg_delay"] > 0 else 0

        # Feature 9: Total execution time
        features["total_duration"] = trace.get("total_duration_ms", sum(durations))

        # Feature 10: Execution rate (actions per second)
        if features["total_duration"] > 0:
            features["execution_rate"] = (len(steps) / features["total_duration"]) * 1000
        else:
            features["execution_rate"] = 0

        # Feature 11-12: Timing anomalies (potential backdoor indicators)
        features["has_long_delays"] = float(any(d > 10000 for d in inter_action_delays))  # >10s delay
        features["has_short_durations"] = float(any(d < 100 for d in durations))  # <100ms duration

        # Feature 13: Timing entropy (regularity of timing)
        features["timing_entropy"] = self._calculate_timing_entropy(durations)

        # Feature 14-15: Burst detection
        features["has_burst"] = float(self._detect_burst(inter_action_delays))
        features["burst_intensity"] = self._calculate_burst_intensity(inter_action_delays)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            "avg_duration",
            "max_duration",
            "std_duration",
            "duration_variation",
            "avg_delay",
            "max_delay",
            "std_delay",
            "delay_variation",
            "total_duration",
            "execution_rate",
            "has_long_delays",
            "has_short_durations",
            "timing_entropy",
            "has_burst",
            "burst_intensity",
        ]

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features for traces with no steps."""
        return {name: 0.0 for name in self.get_feature_names()}

    def _calculate_timing_entropy(self, durations: List[float]) -> float:
        """Calculate entropy of timing distribution."""
        if not durations:
            return 0.0

        # Bin durations into categories
        bins = np.histogram(durations, bins=10)[0]
        total = sum(bins)

        if total == 0:
            return 0.0

        probs = [b / total for b in bins if b > 0]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return entropy

    def _detect_burst(self, delays: List[float]) -> bool:
        """
        Detect burst patterns (multiple actions in quick succession).
        Returns True if there's a burst of at least 3 actions with <200ms delays.
        """
        if len(delays) < 2:
            return False

        burst_count = 0
        for delay in delays:
            if delay < 200:  # 200ms threshold
                burst_count += 1
                if burst_count >= 2:  # 3 consecutive actions
                    return True
            else:
                burst_count = 0

        return False

    def _calculate_burst_intensity(self, delays: List[float]) -> float:
        """
        Calculate burst intensity (ratio of short delays).
        """
        if not delays:
            return 0.0

        short_delays = sum(1 for d in delays if d < 200)
        return short_delays / len(delays)
