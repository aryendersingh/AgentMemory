"""Base memory class that all memory types inherit from."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from agent_memory.core.memory import MemoryQuery, MemoryRecord, ScoredMemory
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType


class BaseMemory(ABC):
    """Abstract base class for all memory types.

    Provides common functionality for storing, retrieving, and
    querying memories through a pluggable storage backend.
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent this memory belongs to.
            embedding_provider: Optional embedding provider for semantic search.
        """
        self._backend = backend
        self._agent_id = agent_id
        self._embedding_provider = embedding_provider

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        ...

    @property
    def agent_id(self) -> str:
        """The ID of the agent this memory belongs to."""
        return self._agent_id

    def _generate_id(self) -> MemoryId:
        """Generate a unique memory ID."""
        return MemoryId(str(uuid.uuid4()))

    async def _create_record(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        ttl: timedelta | None = None,
        generate_embedding: bool = False,
    ) -> MemoryRecord:
        """Create a new memory record.

        Args:
            content: The text content of the memory.
            importance: Importance score from 0.0 to 1.0.
            metadata: Optional metadata dictionary.
            ttl: Optional time-to-live for expiration.
            generate_embedding: Whether to generate an embedding.

        Returns:
            A new MemoryRecord instance.
        """
        embedding = None
        if generate_embedding and self._embedding_provider is not None:
            embedding = await self._embedding_provider.embed(content)

        now = datetime.utcnow()
        return MemoryRecord(
            id=self._generate_id(),
            content=content,
            memory_type=self.memory_type,
            importance=importance,
            created_at=now,
            accessed_at=now,
            access_count=0,
            metadata=metadata or {},
            embedding=embedding,
            agent_id=self._agent_id,
            ttl=ttl,
        )

    async def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        ttl: timedelta | None = None,
    ) -> MemoryId:
        """Add a new memory.

        Args:
            content: The text content of the memory.
            importance: Importance score from 0.0 to 1.0.
            metadata: Optional metadata dictionary.
            ttl: Optional time-to-live for expiration.

        Returns:
            The ID of the stored memory.
        """
        record = await self._create_record(
            content=content,
            importance=importance,
            metadata=metadata,
            ttl=ttl,
        )
        return await self._backend.store(record)

    async def get(self, memory_id: MemoryId) -> MemoryRecord | None:
        """Get a memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The memory record if found, None otherwise.
        """
        return await self._backend.retrieve(memory_id)

    async def remove(self, memory_id: MemoryId) -> bool:
        """Remove a memory by its ID.

        Args:
            memory_id: The ID of the memory to remove.

        Returns:
            True if removed, False if not found.
        """
        return await self._backend.delete(memory_id)

    async def list(
        self,
        limit: int = 100,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
    ) -> Sequence[MemoryRecord]:
        """List memories of this type.

        Args:
            limit: Maximum number of results.
            min_importance: Minimum importance threshold.
            tags: Optional filter by tags.

        Returns:
            Sequence of matching memory records.
        """
        query = MemoryQuery(
            agent_id=self._agent_id,
            memory_type=self.memory_type,
            min_importance=min_importance,
            tags=tags,
            limit=limit,
        )
        return await self._backend.query(query)

    async def search(
        self,
        text: str | None = None,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> Sequence[MemoryRecord]:
        """Search memories by text content or tags.

        Args:
            text: Text to search for in content.
            tags: Tags to filter by.
            min_importance: Minimum importance threshold.
            limit: Maximum number of results.

        Returns:
            Sequence of matching memory records.
        """
        query = MemoryQuery(
            agent_id=self._agent_id,
            memory_type=self.memory_type,
            content_contains=text,
            tags=tags,
            min_importance=min_importance,
            limit=limit,
        )
        return await self._backend.query(query)

    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> Sequence[ScoredMemory]:
        """Search for semantically similar memories.

        Args:
            query: The text query to search for.
            limit: Maximum number of results.
            min_score: Minimum similarity score.

        Returns:
            Sequence of scored memories.

        Raises:
            ValueError: If no embedding provider is configured.
        """
        if self._embedding_provider is None:
            raise ValueError("No embedding provider configured for semantic search")

        embedding = await self._embedding_provider.embed(query)
        return await self._backend.search_similar(
            embedding=embedding,
            limit=limit,
            min_score=min_score,
            memory_type=self.memory_type.value,
            agent_id=self._agent_id,
        )

    async def clear(self) -> int:
        """Clear all memories of this type for this agent.

        Returns:
            Number of memories cleared.
        """
        # Query all memories of this type
        all_memories = await self.list(limit=10000)
        count = 0
        for memory in all_memories:
            if await self._backend.delete(memory.id):
                count += 1
        return count

    async def count(self) -> int:
        """Count memories of this type.

        Returns:
            Number of memories.
        """
        memories = await self.list(limit=100000)
        return len(memories)
