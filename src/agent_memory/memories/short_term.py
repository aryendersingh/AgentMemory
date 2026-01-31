"""Short-term memory for session and task context."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from agent_memory.core.memory import MemoryQuery, MemoryRecord
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory


class ShortTermMemory(BaseMemory):
    """Short-term memory for current session/task context.

    Short-term memories are ephemeral and automatically expire after
    a configurable time period. Use this for:
    - Current task context
    - Session-specific information
    - Temporary working memory
    """

    DEFAULT_TTL = timedelta(hours=1)

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
        default_ttl: timedelta | None = None,
    ) -> None:
        """Initialize short-term memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: Optional embedding provider.
            default_ttl: Default time-to-live for memories (default: 1 hour).
        """
        super().__init__(backend, agent_id, embedding_provider)
        self._default_ttl = default_ttl or self.DEFAULT_TTL

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.SHORT_TERM

    async def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        ttl: timedelta | None = None,
    ) -> MemoryId:
        """Add a short-term memory.

        Args:
            content: The text content of the memory.
            importance: Importance score from 0.0 to 1.0.
            metadata: Optional metadata dictionary.
            ttl: Time-to-live (defaults to default_ttl).

        Returns:
            The ID of the stored memory.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        record = await self._create_record(
            content=content,
            importance=importance,
            metadata=metadata,
            ttl=effective_ttl,
        )
        return await self._backend.store(record)

    async def add_context(
        self,
        context: str,
        importance: float = 0.6,
        context_type: str = "general",
    ) -> MemoryId:
        """Add context information for the current task/session.

        Args:
            context: The context information.
            importance: Importance score.
            context_type: Type of context (e.g., "user_preference", "task_state").

        Returns:
            The ID of the stored memory.
        """
        return await self.add(
            content=context,
            importance=importance,
            metadata={"context_type": context_type},
        )

    async def get_context(
        self,
        context_type: str | None = None,
        limit: int = 10,
    ) -> Sequence[MemoryRecord]:
        """Get recent context memories.

        Args:
            context_type: Optional filter by context type.
            limit: Maximum number of results.

        Returns:
            Sequence of context memories.
        """
        memories = await self.list(limit=limit)

        if context_type is not None:
            return [
                m for m in memories
                if m.metadata.get("context_type") == context_type
            ]

        return memories

    async def set_working_memory(
        self,
        key: str,
        value: str,
        importance: float = 0.7,
    ) -> MemoryId:
        """Set a keyed working memory value.

        This is useful for storing temporary state that needs
        to be referenced by key.

        Args:
            key: The key for this working memory.
            value: The value to store.
            importance: Importance score.

        Returns:
            The ID of the stored memory.
        """
        # Remove any existing memory with this key
        existing = await self.get_working_memory(key)
        if existing is not None:
            await self.remove(existing.id)

        return await self.add(
            content=value,
            importance=importance,
            metadata={"working_memory_key": key},
        )

    async def get_working_memory(self, key: str) -> MemoryRecord | None:
        """Get a working memory value by key.

        Args:
            key: The key to look up.

        Returns:
            The memory record if found, None otherwise.
        """
        memories = await self.list(limit=1000)
        for memory in memories:
            if memory.metadata.get("working_memory_key") == key:
                return memory
        return None

    async def cleanup_expired(self) -> int:
        """Remove expired short-term memories.

        Returns:
            Number of memories removed.
        """
        # Get all memories including expired ones
        query = MemoryQuery(
            agent_id=self._agent_id,
            memory_type=self.memory_type,
            include_expired=True,
            limit=10000,
        )
        all_memories = await self._backend.query(query)

        count = 0
        for memory in all_memories:
            if memory.is_expired():
                if await self._backend.delete(memory.id):
                    count += 1

        return count

    async def extend_ttl(
        self,
        memory_id: MemoryId,
        additional_time: timedelta | None = None,
    ) -> bool:
        """Extend the TTL of a memory.

        Args:
            memory_id: The ID of the memory to extend.
            additional_time: Additional time to add (defaults to default_ttl).

        Returns:
            True if extended, False if memory not found.
        """
        memory = await self._backend.retrieve(memory_id)
        if memory is None:
            return False

        extension = additional_time or self._default_ttl
        new_ttl = (memory.ttl or timedelta(0)) + extension

        # Create updated record with new TTL
        updated = MemoryRecord(
            id=memory.id,
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            created_at=memory.created_at,
            accessed_at=memory.accessed_at,
            access_count=memory.access_count,
            metadata=memory.metadata,
            embedding=memory.embedding,
            agent_id=memory.agent_id,
            ttl=new_ttl,
        )

        return await self._backend.update(updated)
