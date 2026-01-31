"""Importance scoring strategies for memories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Protocol

from agent_memory.core.memory import MemoryRecord


class ImportanceScorer(Protocol):
    """Protocol for importance scoring functions."""

    def __call__(self, record: MemoryRecord) -> float:
        """Calculate the importance score for a memory.

        Args:
            record: The memory record to score.

        Returns:
            Importance score between 0.0 and 1.0.
        """
        ...


def calculate_importance(
    record: MemoryRecord,
    base_weight: float = 0.4,
    access_weight: float = 0.3,
    recency_weight: float = 0.3,
    recency_window: timedelta = timedelta(days=7),
) -> float:
    """Calculate composite importance score.

    Combines base importance, access frequency, and recency
    into a single score.

    Args:
        record: The memory record to score.
        base_weight: Weight for the original importance value.
        access_weight: Weight for access frequency component.
        recency_weight: Weight for recency component.
        recency_window: Time window for recency calculation.

    Returns:
        Combined importance score between 0.0 and 1.0.
    """
    # Base importance
    base_score = record.importance

    # Access frequency component (normalize access count)
    # Using log scale to prevent runaway scores
    import math

    access_score = min(1.0, math.log10(record.access_count + 1) / 3)

    # Recency component
    recency_score = recency_boost(record, window=recency_window)

    # Weighted combination
    total = (
        base_weight * base_score
        + access_weight * access_score
        + recency_weight * recency_score
    )

    return min(1.0, max(0.0, total))


def recency_boost(
    record: MemoryRecord,
    window: timedelta = timedelta(days=7),
) -> float:
    """Calculate recency boost for a memory.

    Returns 1.0 for memories accessed just now, decreasing
    to 0.0 for memories older than the window.

    Args:
        record: The memory record.
        window: Time window for boost calculation.

    Returns:
        Recency score between 0.0 and 1.0.
    """
    now = datetime.utcnow()
    age = now - record.accessed_at

    if age >= window:
        return 0.0

    # Linear decay within the window
    return 1.0 - (age.total_seconds() / window.total_seconds())


def access_frequency_score(
    record: MemoryRecord,
    max_accesses: int = 100,
) -> float:
    """Calculate score based on access frequency.

    Args:
        record: The memory record.
        max_accesses: Access count that corresponds to max score.

    Returns:
        Frequency score between 0.0 and 1.0.
    """
    return min(1.0, record.access_count / max_accesses)


def create_keyword_importance_scorer(
    keywords: dict[str, float],
    base_importance: float = 0.5,
) -> ImportanceScorer:
    """Create a scorer that boosts importance for keyword matches.

    Args:
        keywords: Dictionary mapping keywords to importance boosts.
        base_importance: Default importance for non-matching content.

    Returns:
        A scoring function.
    """

    def scorer(record: MemoryRecord) -> float:
        content_lower = record.content.lower()
        boost = 0.0

        for keyword, keyword_boost in keywords.items():
            if keyword.lower() in content_lower:
                boost = max(boost, keyword_boost)

        return min(1.0, base_importance + boost)

    return scorer


def create_tag_importance_scorer(
    tag_weights: dict[str, float],
    default_weight: float = 0.5,
) -> ImportanceScorer:
    """Create a scorer based on memory tags.

    Args:
        tag_weights: Dictionary mapping tags to importance values.
        default_weight: Default importance for untagged memories.

    Returns:
        A scoring function.
    """

    def scorer(record: MemoryRecord) -> float:
        tags = record.metadata.get("tags", [])
        if not tags:
            return default_weight

        # Use the maximum tag weight
        max_weight = max(
            tag_weights.get(tag, default_weight)
            for tag in tags
        )
        return max_weight

    return scorer


class CompositeImportanceScorer:
    """Combine multiple importance scorers."""

    def __init__(
        self,
        scorers: list[tuple[ImportanceScorer, float]],
    ) -> None:
        """Initialize with weighted scorers.

        Args:
            scorers: List of (scorer, weight) tuples.
        """
        self._scorers = scorers
        total_weight = sum(weight for _, weight in scorers)
        self._normalized_weights = [
            (scorer, weight / total_weight)
            for scorer, weight in scorers
        ]

    def __call__(self, record: MemoryRecord) -> float:
        """Calculate composite importance score.

        Args:
            record: The memory record to score.

        Returns:
            Weighted average of all scorer outputs.
        """
        total = sum(
            scorer(record) * weight
            for scorer, weight in self._normalized_weights
        )
        return min(1.0, max(0.0, total))
