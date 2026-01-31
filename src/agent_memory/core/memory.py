"""Memory record and query dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from agent_memory.core.types import MemoryId, MemoryType


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """An immutable memory record stored in the memory system.

    Attributes:
        id: Unique identifier for this memory.
        content: The text content of the memory.
        memory_type: The type/category of this memory.
        importance: Importance score from 0.0 to 1.0.
        created_at: When the memory was created.
        accessed_at: When the memory was last accessed.
        access_count: Number of times this memory has been accessed.
        metadata: Additional metadata as key-value pairs.
        embedding: Optional vector embedding for semantic search.
        agent_id: The ID of the agent this memory belongs to.
        ttl: Optional time-to-live for automatic expiration.
    """

    id: MemoryId
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    agent_id: str = "default"
    ttl: timedelta | None = None

    def __post_init__(self) -> None:
        """Validate the memory record after initialization."""
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"Importance must be between 0.0 and 1.0, got {self.importance}")

    def with_access(self) -> "MemoryRecord":
        """Return a new record with updated access time and count."""
        return MemoryRecord(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            importance=self.importance,
            created_at=self.created_at,
            accessed_at=datetime.utcnow(),
            access_count=self.access_count + 1,
            metadata=self.metadata,
            embedding=self.embedding,
            agent_id=self.agent_id,
            ttl=self.ttl,
        )

    def with_importance(self, importance: float) -> "MemoryRecord":
        """Return a new record with updated importance score."""
        return MemoryRecord(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            importance=importance,
            created_at=self.created_at,
            accessed_at=self.accessed_at,
            access_count=self.access_count,
            metadata=self.metadata,
            embedding=self.embedding,
            agent_id=self.agent_id,
            ttl=self.ttl,
        )

    def with_embedding(self, embedding: list[float]) -> "MemoryRecord":
        """Return a new record with an embedding vector."""
        return MemoryRecord(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            importance=self.importance,
            created_at=self.created_at,
            accessed_at=self.accessed_at,
            access_count=self.access_count,
            metadata=self.metadata,
            embedding=embedding,
            agent_id=self.agent_id,
            ttl=self.ttl,
        )

    def with_metadata(self, **kwargs: Any) -> "MemoryRecord":
        """Return a new record with updated metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return MemoryRecord(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            importance=self.importance,
            created_at=self.created_at,
            accessed_at=self.accessed_at,
            access_count=self.access_count,
            metadata=new_metadata,
            embedding=self.embedding,
            agent_id=self.agent_id,
            ttl=self.ttl,
        )

    def is_expired(self) -> bool:
        """Check if this memory has expired based on its TTL."""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + self.ttl


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    """Query parameters for searching memories.

    Attributes:
        agent_id: Filter by agent ID.
        memory_type: Filter by memory type.
        min_importance: Minimum importance threshold.
        tags: Filter by metadata tags.
        content_contains: Text substring to search for.
        created_after: Filter memories created after this time.
        created_before: Filter memories created before this time.
        limit: Maximum number of results to return.
        offset: Number of results to skip.
        include_expired: Whether to include expired memories.
    """

    agent_id: str | None = None
    memory_type: MemoryType | None = None
    min_importance: float = 0.0
    tags: list[str] | None = None
    content_contains: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100
    offset: int = 0
    include_expired: bool = False


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A memory record with an associated relevance score.

    Used for search results where memories are ranked by similarity
    or other scoring criteria.
    """

    memory: MemoryRecord
    score: float

    def __lt__(self, other: "ScoredMemory") -> bool:
        """Compare by score for sorting (higher scores first)."""
        return self.score > other.score
