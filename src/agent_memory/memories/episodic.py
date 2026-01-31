"""Episodic memory for events and experiences."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agent_memory.core.memory import MemoryQuery, MemoryRecord
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory


@dataclass
class Episode:
    """A structured episode representation."""

    event_type: str
    description: str
    outcome: str | None
    participants: list[str]
    location: str | None
    timestamp: datetime
    duration: timedelta | None
    metadata: dict[str, Any]


class EpisodicMemory(BaseMemory):
    """Memory for events, experiences, and episodes.

    Episodic memory stores temporally-ordered events with rich
    contextual information. Use this for:
    - Recording actions and their outcomes
    - Tracking user interactions
    - Storing notable events
    - Building experiential knowledge
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize episodic memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: Optional embedding provider.
        """
        super().__init__(backend, agent_id, embedding_provider)

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.EPISODIC

    async def record_event(
        self,
        event_type: str,
        description: str,
        outcome: str | None = None,
        participants: list[str] | None = None,
        location: str | None = None,
        importance: float = 0.6,
        duration: timedelta | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> MemoryId:
        """Record an event/episode.

        Args:
            event_type: Type of event (e.g., "task_completed", "error_occurred").
            description: Description of what happened.
            outcome: Result of the event (e.g., "success", "failure").
            participants: List of participants involved.
            location: Optional location/context.
            importance: Importance score.
            duration: How long the event took.
            extra_metadata: Additional metadata.

        Returns:
            The ID of the stored episode.
        """
        metadata: dict[str, Any] = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if outcome:
            metadata["outcome"] = outcome
        if participants:
            metadata["participants"] = participants
        if location:
            metadata["location"] = location
        if duration:
            metadata["duration_seconds"] = duration.total_seconds()
        if extra_metadata:
            metadata.update(extra_metadata)

        return await self.add(
            content=description,
            importance=importance,
            metadata=metadata,
            ttl=None,  # Episodes are permanent by default
        )

    async def record_action(
        self,
        action: str,
        result: str,
        success: bool = True,
        importance: float = 0.5,
    ) -> MemoryId:
        """Record an action and its result.

        Args:
            action: The action taken.
            result: The result of the action.
            success: Whether the action was successful.
            importance: Importance score.

        Returns:
            The ID of the stored episode.
        """
        return await self.record_event(
            event_type="action",
            description=f"Action: {action}\nResult: {result}",
            outcome="success" if success else "failure",
            importance=importance,
        )

    async def get_events_by_type(
        self,
        event_type: str,
        limit: int = 50,
    ) -> Sequence[MemoryRecord]:
        """Get events of a specific type.

        Args:
            event_type: The type of events to retrieve.
            limit: Maximum number of results.

        Returns:
            Sequence of matching event memories.
        """
        all_events = await self.list(limit=limit * 2)
        return [
            e for e in all_events
            if e.metadata.get("event_type") == event_type
        ][:limit]

    async def get_events_in_range(
        self,
        start: datetime,
        end: datetime | None = None,
        limit: int = 100,
    ) -> Sequence[MemoryRecord]:
        """Get events within a time range.

        Args:
            start: Start of the time range.
            end: End of the time range (defaults to now).
            limit: Maximum number of results.

        Returns:
            Sequence of events in the time range.
        """
        end = end or datetime.utcnow()

        query = MemoryQuery(
            agent_id=self._agent_id,
            memory_type=self.memory_type,
            created_after=start,
            created_before=end,
            limit=limit,
        )

        return await self._backend.query(query)

    async def get_recent_events(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> Sequence[MemoryRecord]:
        """Get events from the last N hours.

        Args:
            hours: Number of hours to look back.
            limit: Maximum number of results.

        Returns:
            Sequence of recent events.
        """
        start = datetime.utcnow() - timedelta(hours=hours)
        return await self.get_events_in_range(start=start, limit=limit)

    async def get_successful_events(
        self,
        event_type: str | None = None,
        limit: int = 50,
    ) -> Sequence[MemoryRecord]:
        """Get events with successful outcomes.

        Args:
            event_type: Optional event type filter.
            limit: Maximum number of results.

        Returns:
            Sequence of successful events.
        """
        all_events = await self.list(limit=limit * 2)

        successful = [
            e for e in all_events
            if e.metadata.get("outcome") == "success"
        ]

        if event_type:
            successful = [
                e for e in successful
                if e.metadata.get("event_type") == event_type
            ]

        return successful[:limit]

    async def get_failed_events(
        self,
        event_type: str | None = None,
        limit: int = 50,
    ) -> Sequence[MemoryRecord]:
        """Get events with failed outcomes.

        Args:
            event_type: Optional event type filter.
            limit: Maximum number of results.

        Returns:
            Sequence of failed events.
        """
        all_events = await self.list(limit=limit * 2)

        failed = [
            e for e in all_events
            if e.metadata.get("outcome") == "failure"
        ]

        if event_type:
            failed = [
                e for e in failed
                if e.metadata.get("event_type") == event_type
            ]

        return failed[:limit]

    async def find_similar_events(
        self,
        description: str,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> Sequence[MemoryRecord]:
        """Find events similar to a description.

        Args:
            description: Description to match against.
            limit: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            Sequence of similar events.

        Raises:
            ValueError: If no embedding provider is configured.
        """
        scored = await self.search_similar(
            query=description,
            limit=limit,
            min_score=min_score,
        )
        return [s.memory for s in scored]

    async def get_event_timeline(
        self,
        limit: int = 20,
    ) -> list[Episode]:
        """Get a timeline of recent events as Episode objects.

        Args:
            limit: Maximum number of events.

        Returns:
            List of Episode objects.
        """
        events = await self.list(limit=limit)

        timeline = []
        for event in sorted(events, key=lambda e: e.created_at):
            timeline.append(
                Episode(
                    event_type=event.metadata.get("event_type", "unknown"),
                    description=event.content,
                    outcome=event.metadata.get("outcome"),
                    participants=event.metadata.get("participants", []),
                    location=event.metadata.get("location"),
                    timestamp=event.created_at,
                    duration=(
                        timedelta(seconds=event.metadata["duration_seconds"])
                        if "duration_seconds" in event.metadata
                        else None
                    ),
                    metadata={
                        k: v
                        for k, v in event.metadata.items()
                        if k
                        not in {
                            "event_type",
                            "outcome",
                            "participants",
                            "location",
                            "duration_seconds",
                            "timestamp",
                        }
                    },
                )
            )

        return timeline
