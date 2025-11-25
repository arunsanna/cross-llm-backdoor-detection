"""
Sequence-level feature extraction.

Captures patterns in action sequences:
- Action transitions and n-grams
- Sequence anomalies
- Repetitive patterns
- Unusual action orderings
"""

from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
from .base import BaseFeatureExtractor


class SequenceFeatureExtractor(BaseFeatureExtractor):
    """Extract sequence-level behavioral features."""

    def __init__(self):
        super().__init__("sequence_level")

    def extract(self, trace: Dict[str, Any]) -> Dict[str, float]:
        """Extract sequence-level features from a trace."""
        steps = trace.get("steps", [])

        if len(steps) < 2:
            return self._empty_features()

        features = {}

        # Extract tool sequence (handle tool as dict or string)
        tool_sequence = []
        for step in steps:
            tool = step.get("tool", "unknown")
            if isinstance(tool, dict):
                tool = tool.get("name", "unknown")
            tool_sequence.append(tool)

        # Feature 1-3: Bigram (2-gram) statistics
        bigrams = list(zip(tool_sequence[:-1], tool_sequence[1:]))
        bigram_counts = Counter(bigrams)
        features["unique_bigrams"] = len(bigram_counts)
        features["max_bigram_freq"] = max(bigram_counts.values()) if bigram_counts else 0
        features["bigram_diversity"] = len(bigram_counts) / len(bigrams) if bigrams else 0

        # Feature 4-6: Trigram (3-gram) statistics
        if len(tool_sequence) >= 3:
            trigrams = list(zip(tool_sequence[:-2], tool_sequence[1:-1], tool_sequence[2:]))
            trigram_counts = Counter(trigrams)
            features["unique_trigrams"] = len(trigram_counts)
            features["trigram_diversity"] = len(trigram_counts) / len(trigrams) if trigrams else 0
        else:
            features["unique_trigrams"] = 0
            features["trigram_diversity"] = 0

        # Feature 7-8: Repetition patterns (backdoor indicator)
        features["repetition_ratio"] = self._calculate_repetition_ratio(tool_sequence)
        features["max_consecutive_repeats"] = self._max_consecutive_repeats(tool_sequence)

        # Feature 9-10: Sequence length and complexity
        features["sequence_length"] = len(tool_sequence)
        features["unique_transitions"] = len(set(bigrams))

        # Feature 11: Markov chain entropy (sequence unpredictability)
        features["transition_entropy"] = self._calculate_transition_entropy(bigrams)

        # Feature 12-13: Loop detection
        features["has_loops"] = float(self._has_loops(tool_sequence))
        features["loop_count"] = self._count_loops(tool_sequence)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            "unique_bigrams",
            "max_bigram_freq",
            "bigram_diversity",
            "unique_trigrams",
            "trigram_diversity",
            "repetition_ratio",
            "max_consecutive_repeats",
            "sequence_length",
            "unique_transitions",
            "transition_entropy",
            "has_loops",
            "loop_count",
        ]

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features for short sequences."""
        return {name: 0.0 for name in self.get_feature_names()}

    def _calculate_repetition_ratio(self, sequence: List[str]) -> float:
        """Calculate ratio of repeated elements."""
        if not sequence:
            return 0.0
        unique = len(set(sequence))
        return 1.0 - (unique / len(sequence))

    def _max_consecutive_repeats(self, sequence: List[str]) -> int:
        """Find maximum consecutive repetitions of same element."""
        if not sequence:
            return 0

        max_repeats = 1
        current_repeats = 1

        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current_repeats += 1
                max_repeats = max(max_repeats, current_repeats)
            else:
                current_repeats = 1

        return max_repeats

    def _calculate_transition_entropy(self, bigrams: List[Tuple[str, str]]) -> float:
        """Calculate entropy of state transitions."""
        if not bigrams:
            return 0.0

        counts = Counter(bigrams)
        total = len(bigrams)
        probs = [count / total for count in counts.values()]

        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def _has_loops(self, sequence: List[str]) -> bool:
        """Detect if sequence contains loops (repeated subsequences)."""
        for length in range(2, len(sequence) // 2 + 1):
            for i in range(len(sequence) - length * 2 + 1):
                subseq = sequence[i:i + length]
                if subseq == sequence[i + length:i + length * 2]:
                    return True
        return False

    def _count_loops(self, sequence: List[str]) -> int:
        """Count number of loops in sequence."""
        loop_count = 0
        checked = set()

        for length in range(2, len(sequence) // 2 + 1):
            for i in range(len(sequence) - length * 2 + 1):
                if i in checked:
                    continue
                subseq = tuple(sequence[i:i + length])
                if subseq == tuple(sequence[i + length:i + length * 2]):
                    loop_count += 1
                    checked.update(range(i, i + length * 2))

        return loop_count
