"""Long-term memory for persistent facts and knowledge."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from agent_memory.core.memory import MemoryQuery, MemoryRecord
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory


class LongTermMemory(BaseMemory):
    """Long-term memory for persistent facts, preferences, and knowledge.

    Long-term memories do not expire and are meant to persist across
    sessions. Use this for:
    - User preferences and settings
    - Learned facts about the user
    - Domain knowledge
    - Persistent state
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize long-term memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: Optional embedding provider.
        """
        super().__init__(backend, agent_id, embedding_provider)

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.LONG_TERM

    async def store_fact(
        self,
        fact: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> MemoryId:
        """Store a fact in long-term memory.

        Args:
            fact: The fact to store.
            importance: Importance score.
            tags: Optional tags for categorization.
            source: Optional source of the fact.

        Returns:
            The ID of the stored memory.
        """
        metadata: dict[str, Any] = {"fact": True}
        if tags:
            metadata["tags"] = tags
        if source:
            metadata["source"] = source

        return await self.add(
            content=fact,
            importance=importance,
            metadata=metadata,
            ttl=None,  # Long-term memories don't expire
        )

    async def store_preference(
        self,
        category: str,
        preference: str,
        importance: float = 0.8,
    ) -> MemoryId:
        """Store a user preference.

        Args:
            category: Category of the preference (e.g., "ui", "communication").
            preference: The preference value.
            importance: Importance score.

        Returns:
            The ID of the stored memory.
        """
        # Check for existing preference in same category
        existing = await self.get_preferences(category)
        for pref in existing:
            # If preference already exists with same content, update importance
            if pref.content == preference:
                updated = pref.with_importance(max(pref.importance, importance))
                await self._backend.update(updated)
                return pref.id

        return await self.add(
            content=preference,
            importance=importance,
            metadata={
                "preference": True,
                "category": category,
            },
            ttl=None,
        )

    async def get_preferences(
        self,
        category: str | None = None,
    ) -> Sequence[MemoryRecord]:
        """Get stored preferences.

        Args:
            category: Optional category filter.

        Returns:
            Sequence of preference memories.
        """
        memories = await self.list(limit=1000)

        preferences = [
            m for m in memories
            if m.metadata.get("preference") is True
        ]

        if category is not None:
            preferences = [
                p for p in preferences
                if p.metadata.get("category") == category
            ]

        return preferences

    async def get_facts(
        self,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> Sequence[MemoryRecord]:
        """Get stored facts.

        Args:
            tags: Optional tags to filter by.
            limit: Maximum number of results.

        Returns:
            Sequence of fact memories.
        """
        if tags:
            memories = await self.search(tags=tags, limit=limit)
        else:
            memories = await self.list(limit=limit)

        return [m for m in memories if m.metadata.get("fact") is True]

    async def update_fact(
        self,
        memory_id: MemoryId,
        new_content: str,
        updated_importance: float | None = None,
    ) -> bool:
        """Update an existing fact.

        Args:
            memory_id: The ID of the fact to update.
            new_content: The new content for the fact.
            updated_importance: Optional new importance score.

        Returns:
            True if updated, False if not found.
        """
        memory = await self._backend.retrieve(memory_id)
        if memory is None:
            return False

        updated = MemoryRecord(
            id=memory.id,
            content=new_content,
            memory_type=memory.memory_type,
            importance=updated_importance if updated_importance is not None else memory.importance,
            created_at=memory.created_at,
            accessed_at=datetime.utcnow(),
            access_count=memory.access_count + 1,
            metadata={**memory.metadata, "updated_at": datetime.utcnow().isoformat()},
            embedding=None,  # Clear embedding since content changed
            agent_id=memory.agent_id,
            ttl=memory.ttl,
        )

        return await self._backend.update(updated)

    async def consolidate_duplicates(
        self,
        similarity_threshold: float = 0.9,
    ) -> int:
        """Find and merge duplicate or very similar memories.

        Args:
            similarity_threshold: Minimum similarity score to consider duplicates.

        Returns:
            Number of memories merged.
        """
        if self._embedding_provider is None:
            return 0

        memories = await self.list(limit=10000)
        merged_count = 0
        processed_ids: set[MemoryId] = set()

        for memory in memories:
            if memory.id in processed_ids:
                continue

            # Get embedding for this memory
            if memory.embedding is None:
                embedding = await self._embedding_provider.embed(memory.content)
            else:
                embedding = memory.embedding

            # Find similar memories
            similar = await self._backend.search_similar(
                embedding=embedding,
                limit=10,
                min_score=similarity_threshold,
                memory_type=self.memory_type.value,
                agent_id=self._agent_id,
            )

            # Merge duplicates (keep the one with highest importance)
            duplicates = [
                s.memory for s in similar
                if s.memory.id != memory.id and s.memory.id not in processed_ids
            ]

            if duplicates:
                # Keep the highest importance memory
                all_versions = [memory] + list(duplicates)
                keeper = max(all_versions, key=lambda m: m.importance)

                # Delete the others
                for dup in all_versions:
                    if dup.id != keeper.id:
                        await self._backend.delete(dup.id)
                        processed_ids.add(dup.id)
                        merged_count += 1

            processed_ids.add(memory.id)

        return merged_count

    async def get_by_tags(self, tags: list[str]) -> Sequence[MemoryRecord]:
        """Get memories by tags.

        Args:
            tags: Tags to filter by (any match).

        Returns:
            Sequence of matching memories.
        """
        return await self.search(tags=tags)

    async def add_tags(self, memory_id: MemoryId, tags: list[str]) -> bool:
        """Add tags to an existing memory.

        Args:
            memory_id: The ID of the memory.
            tags: Tags to add.

        Returns:
            True if updated, False if not found.
        """
        memory = await self._backend.retrieve(memory_id)
        if memory is None:
            return False

        existing_tags = memory.metadata.get("tags", [])
        new_tags = list(set(existing_tags + tags))

        updated = memory.with_metadata(tags=new_tags)
        return await self._backend.update(updated)
