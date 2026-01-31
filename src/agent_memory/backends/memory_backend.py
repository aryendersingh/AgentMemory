"""In-memory storage backend implementation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from agent_memory.core.memory import MemoryQuery, MemoryRecord, ScoredMemory
from agent_memory.core.types import MemoryId


class InMemoryBackend:
    """Reference implementation of a storage backend using in-memory dict storage.

    This backend is suitable for development, testing, and small-scale
    applications. Data is not persisted across restarts.
    """

    def __init__(self) -> None:
        """Initialize the in-memory storage."""
        self._storage: dict[MemoryId, MemoryRecord] = {}

    async def store(self, record: MemoryRecord) -> MemoryId:
        """Store a memory record.

        Args:
            record: The memory record to store.

        Returns:
            The ID of the stored record.
        """
        self._storage[record.id] = record
        return record.id

    async def retrieve(self, memory_id: MemoryId) -> MemoryRecord | None:
        """Retrieve a memory record by ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The memory record if found, None otherwise.
        """
        record = self._storage.get(memory_id)
        if record is not None:
            # Update access time and count
            updated = record.with_access()
            self._storage[memory_id] = updated
            return updated
        return None

    async def update(self, record: MemoryRecord) -> bool:
        """Update an existing memory record.

        Args:
            record: The updated memory record.

        Returns:
            True if updated, False if not found.
        """
        if record.id in self._storage:
            self._storage[record.id] = record
            return True
        return False

    async def delete(self, memory_id: MemoryId) -> bool:
        """Delete a memory record.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        if memory_id in self._storage:
            del self._storage[memory_id]
            return True
        return False

    async def query(self, query: MemoryQuery) -> Sequence[MemoryRecord]:
        """Query memories based on filter criteria.

        Args:
            query: Query parameters for filtering.

        Returns:
            Sequence of matching memory records.
        """
        results: list[MemoryRecord] = []

        for record in self._storage.values():
            # Skip expired memories unless explicitly included
            if not query.include_expired and record.is_expired():
                continue

            # Apply filters
            if query.agent_id is not None and record.agent_id != query.agent_id:
                continue

            if query.memory_type is not None and record.memory_type != query.memory_type:
                continue

            if record.importance < query.min_importance:
                continue

            if query.tags is not None:
                record_tags = record.metadata.get("tags", [])
                if not any(tag in record_tags for tag in query.tags):
                    continue

            if query.content_contains is not None:
                if query.content_contains.lower() not in record.content.lower():
                    continue

            if query.created_after is not None and record.created_at < query.created_after:
                continue

            if query.created_before is not None and record.created_at > query.created_before:
                continue

            results.append(record)

        # Sort by creation time (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        start = query.offset
        end = start + query.limit
        return results[start:end]

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        memory_type: str | None = None,
        agent_id: str | None = None,
    ) -> Sequence[ScoredMemory]:
        """Search for similar memories using cosine similarity.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results.
            min_score: Minimum similarity score threshold.
            memory_type: Optional filter by memory type.
            agent_id: Optional filter by agent ID.

        Returns:
            Sequence of scored memories, sorted by similarity.
        """
        scored: list[ScoredMemory] = []

        for record in self._storage.values():
            # Skip records without embeddings
            if record.embedding is None:
                continue

            # Skip expired memories
            if record.is_expired():
                continue

            # Apply filters
            if memory_type is not None and record.memory_type != memory_type:
                continue

            if agent_id is not None and record.agent_id != agent_id:
                continue

            # Calculate cosine similarity
            score = self._cosine_similarity(embedding, record.embedding)

            if score >= min_score:
                scored.append(ScoredMemory(memory=record, score=score))

        # Sort by score (highest first) and limit
        scored.sort()
        return scored[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def clear(self, agent_id: str | None = None) -> int:
        """Clear memories, optionally filtered by agent ID.

        Args:
            agent_id: If provided, only clear memories for this agent.

        Returns:
            Number of memories deleted.
        """
        if agent_id is None:
            count = len(self._storage)
            self._storage.clear()
            return count

        to_delete = [
            mid for mid, record in self._storage.items() if record.agent_id == agent_id
        ]
        for mid in to_delete:
            del self._storage[mid]
        return len(to_delete)

    async def close(self) -> None:
        """Close the backend (no-op for in-memory storage)."""
        pass

    def __len__(self) -> int:
        """Return the number of memories stored."""
        return len(self._storage)

    async def get_stats(self) -> dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        now = datetime.utcnow()
        expired = sum(1 for r in self._storage.values() if r.is_expired())
        by_type: dict[str, int] = {}

        for record in self._storage.values():
            type_name = record.memory_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total": len(self._storage),
            "expired": expired,
            "active": len(self._storage) - expired,
            **by_type,
        }
