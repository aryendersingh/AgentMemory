"""Memory decay functions for time-based importance reduction."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta

from agent_memory.core.memory import MemoryRecord

# Type alias for decay functions
DecayFunction = Callable[[MemoryRecord, datetime], float]


def exponential_decay(
    record: MemoryRecord,
    now: datetime | None = None,
    half_life: timedelta = timedelta(days=7),
) -> float:
    """Apply exponential decay to memory importance.

    The importance halves after each half_life period.

    Args:
        record: The memory record.
        now: Current time (defaults to utcnow).
        half_life: Time for importance to halve.

    Returns:
        Decayed importance value.
    """
    now = now or datetime.utcnow()
    age = now - record.accessed_at

    # Calculate decay factor: 0.5^(age/half_life)
    decay_factor = 0.5 ** (age.total_seconds() / half_life.total_seconds())

    return record.importance * decay_factor


def linear_decay(
    record: MemoryRecord,
    now: datetime | None = None,
    decay_rate: float = 0.1,
    decay_period: timedelta = timedelta(days=1),
    min_importance: float = 0.0,
) -> float:
    """Apply linear decay to memory importance.

    Importance decreases by decay_rate for each decay_period.

    Args:
        record: The memory record.
        now: Current time (defaults to utcnow).
        decay_rate: Amount to decrease per period.
        decay_period: Time period for each decay step.
        min_importance: Minimum importance floor.

    Returns:
        Decayed importance value.
    """
    now = now or datetime.utcnow()
    age = now - record.accessed_at

    periods = age.total_seconds() / decay_period.total_seconds()
    decayed = record.importance - (decay_rate * periods)

    return max(min_importance, decayed)


def stepped_decay(
    record: MemoryRecord,
    now: datetime | None = None,
    steps: list[tuple[timedelta, float]] | None = None,
) -> float:
    """Apply stepped decay based on age thresholds.

    Args:
        record: The memory record.
        now: Current time (defaults to utcnow).
        steps: List of (age_threshold, decay_factor) tuples, sorted by age.
               Default: 1 day=0.9, 1 week=0.7, 1 month=0.5, 3 months=0.3

    Returns:
        Decayed importance value.
    """
    now = now or datetime.utcnow()
    age = now - record.accessed_at

    if steps is None:
        steps = [
            (timedelta(days=1), 0.9),
            (timedelta(days=7), 0.7),
            (timedelta(days=30), 0.5),
            (timedelta(days=90), 0.3),
        ]

    decay_factor = 1.0
    for threshold, factor in steps:
        if age >= threshold:
            decay_factor = factor
        else:
            break

    return record.importance * decay_factor


def no_decay(
    record: MemoryRecord,
    now: datetime | None = None,
) -> float:
    """No decay function - returns original importance.

    Args:
        record: The memory record.
        now: Current time (ignored).

    Returns:
        Original importance value.
    """
    return record.importance


def access_refreshed_decay(
    record: MemoryRecord,
    now: datetime | None = None,
    base_half_life: timedelta = timedelta(days=7),
    access_bonus: timedelta = timedelta(hours=12),
    max_bonus: timedelta = timedelta(days=30),
) -> float:
    """Exponential decay with access-based refresh.

    Each access extends the effective half-life up to a maximum.

    Args:
        record: The memory record.
        now: Current time (defaults to utcnow).
        base_half_life: Base half-life before access bonuses.
        access_bonus: Additional half-life per access.
        max_bonus: Maximum total bonus from accesses.

    Returns:
        Decayed importance value.
    """
    now = now or datetime.utcnow()

    # Calculate effective half-life with access bonus
    bonus = min(
        max_bonus,
        timedelta(seconds=access_bonus.total_seconds() * record.access_count),
    )
    effective_half_life = base_half_life + bonus

    # Apply exponential decay
    age = now - record.accessed_at
    decay_factor = 0.5 ** (age.total_seconds() / effective_half_life.total_seconds())

    return record.importance * decay_factor


def create_custom_decay(
    decay_func: Callable[[float, float], float],
    time_scale: timedelta = timedelta(days=1),
) -> DecayFunction:
    """Create a custom decay function from a mathematical function.

    Args:
        decay_func: Function taking (importance, time_units) and returning decayed value.
        time_scale: What one "time unit" represents.

    Returns:
        A DecayFunction compatible function.

    Example:
        # Logarithmic decay
        log_decay = create_custom_decay(
            lambda imp, t: imp / (1 + math.log(1 + t)),
            timedelta(hours=6)
        )
    """

    def decay(record: MemoryRecord, now: datetime | None = None) -> float:
        now = now or datetime.utcnow()
        age = now - record.accessed_at
        time_units = age.total_seconds() / time_scale.total_seconds()

        result = decay_func(record.importance, time_units)
        return max(0.0, min(1.0, result))

    return decay


class DecayScheduler:
    """Apply different decay functions based on memory type or metadata."""

    def __init__(
        self,
        default_decay: DecayFunction = exponential_decay,
    ) -> None:
        """Initialize with a default decay function.

        Args:
            default_decay: Default decay function to use.
        """
        self._default = default_decay
        self._type_decays: dict[str, DecayFunction] = {}
        self._tag_decays: dict[str, DecayFunction] = {}

    def set_type_decay(self, memory_type: str, decay: DecayFunction) -> None:
        """Set decay function for a memory type.

        Args:
            memory_type: The memory type.
            decay: The decay function to use.
        """
        self._type_decays[memory_type] = decay

    def set_tag_decay(self, tag: str, decay: DecayFunction) -> None:
        """Set decay function for memories with a specific tag.

        Args:
            tag: The tag to match.
            decay: The decay function to use.
        """
        self._tag_decays[tag] = decay

    def apply(self, record: MemoryRecord, now: datetime | None = None) -> float:
        """Apply the appropriate decay function.

        Priority: tag-specific > type-specific > default

        Args:
            record: The memory record.
            now: Current time.

        Returns:
            Decayed importance value.
        """
        now = now or datetime.utcnow()

        # Check tag-specific decay
        tags = record.metadata.get("tags", [])
        for tag in tags:
            if tag in self._tag_decays:
                return self._tag_decays[tag](record, now)

        # Check type-specific decay
        if record.memory_type.value in self._type_decays:
            return self._type_decays[record.memory_type.value](record, now)

        # Fall back to default
        return self._default(record, now)
