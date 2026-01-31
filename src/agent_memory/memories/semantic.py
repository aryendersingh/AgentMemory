"""Semantic memory with vector-based retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent_memory.core.memory import MemoryRecord, ScoredMemory
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory


class SemanticMemory(BaseMemory):
    """Memory with vector-based semantic search.

    Semantic memory stores information with embeddings for
    similarity-based retrieval. Use this for:
    - Knowledge bases
    - Document storage
    - Information that needs semantic search
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """Initialize semantic memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: The embedding provider (required).
        """
        super().__init__(backend, agent_id, embedding_provider)
        if embedding_provider is None:
            raise ValueError("SemanticMemory requires an embedding provider")

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.SEMANTIC

    async def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        ttl: None = None,
    ) -> MemoryId:
        """Add content to semantic memory with auto-generated embedding.

        Args:
            content: The text content to store.
            importance: Importance score.
            metadata: Optional metadata.
            ttl: Not used (semantic memories are permanent).

        Returns:
            The ID of the stored memory.
        """
        record = await self._create_record(
            content=content,
            importance=importance,
            metadata=metadata,
            ttl=None,
            generate_embedding=True,
        )
        return await self._backend.store(record)

    async def add_with_embedding(
        self,
        content: str,
        embedding: list[float],
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryId:
        """Add content with a pre-computed embedding.

        Args:
            content: The text content.
            embedding: Pre-computed embedding vector.
            importance: Importance score.
            metadata: Optional metadata.

        Returns:
            The ID of the stored memory.
        """
        record = await self._create_record(
            content=content,
            importance=importance,
            metadata=metadata,
            ttl=None,
            generate_embedding=False,
        )
        record = record.with_embedding(embedding)
        return await self._backend.store(record)

    async def search(
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
            Sequence of scored memories, sorted by similarity.
        """
        if self._embedding_provider is None:
            raise ValueError("No embedding provider configured")

        embedding = await self._embedding_provider.embed(query)
        return await self._backend.search_similar(
            embedding=embedding,
            limit=limit,
            min_score=min_score,
            memory_type=self.memory_type.value,
            agent_id=self._agent_id,
        )

    async def search_with_filters(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        min_importance: float = 0.0,
    ) -> Sequence[ScoredMemory]:
        """Search with additional filtering.

        Args:
            query: The text query.
            limit: Maximum number of results.
            min_score: Minimum similarity score.
            tags: Optional tags to filter by.
            min_importance: Minimum importance threshold.

        Returns:
            Filtered and scored memories.
        """
        # Get more results than needed for filtering
        results = await self.search(query, limit=limit * 3, min_score=min_score)

        filtered: list[ScoredMemory] = []
        for scored in results:
            memory = scored.memory

            # Apply importance filter
            if memory.importance < min_importance:
                continue

            # Apply tag filter
            if tags:
                memory_tags = memory.metadata.get("tags", [])
                if not any(tag in memory_tags for tag in tags):
                    continue

            filtered.append(scored)

            if len(filtered) >= limit:
                break

        return filtered

    async def find_related(
        self,
        memory_id: MemoryId,
        limit: int = 5,
        min_score: float = 0.5,
    ) -> Sequence[ScoredMemory]:
        """Find memories related to a given memory.

        Args:
            memory_id: The ID of the memory to find relations for.
            limit: Maximum number of related memories.
            min_score: Minimum similarity score.

        Returns:
            Related memories (excluding the original).
        """
        memory = await self._backend.retrieve(memory_id)
        if memory is None:
            return []

        if memory.embedding is None:
            # Generate embedding if not present
            if self._embedding_provider is None:
                return []
            embedding = await self._embedding_provider.embed(memory.content)
        else:
            embedding = memory.embedding

        results = await self._backend.search_similar(
            embedding=embedding,
            limit=limit + 1,  # +1 to exclude self
            min_score=min_score,
            memory_type=self.memory_type.value,
            agent_id=self._agent_id,
        )

        # Filter out the original memory
        return [r for r in results if r.memory.id != memory_id][:limit]

    async def add_document(
        self,
        content: str,
        title: str | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.6,
    ) -> MemoryId:
        """Add a document to semantic memory.

        Args:
            content: The document content.
            title: Optional document title.
            source: Optional source URL or reference.
            tags: Optional tags for categorization.
            importance: Importance score.

        Returns:
            The ID of the stored document.
        """
        metadata: dict[str, Any] = {"document": True}
        if title:
            metadata["title"] = title
        if source:
            metadata["source"] = source
        if tags:
            metadata["tags"] = tags

        return await self.add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    async def add_chunks(
        self,
        chunks: list[str],
        document_id: str | None = None,
        base_metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> list[MemoryId]:
        """Add multiple text chunks (e.g., from a split document).

        Args:
            chunks: List of text chunks.
            document_id: Optional ID linking chunks to parent document.
            base_metadata: Metadata to apply to all chunks.
            importance: Importance score for all chunks.

        Returns:
            List of stored memory IDs.
        """
        ids: list[MemoryId] = []

        for i, chunk in enumerate(chunks):
            metadata: dict[str, Any] = {
                "chunk": True,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if document_id:
                metadata["document_id"] = document_id
            if base_metadata:
                metadata.update(base_metadata)

            memory_id = await self.add(
                content=chunk,
                importance=importance,
                metadata=metadata,
            )
            ids.append(memory_id)

        return ids

    async def get_by_document_id(
        self,
        document_id: str,
    ) -> Sequence[MemoryRecord]:
        """Get all chunks for a document.

        Args:
            document_id: The document ID.

        Returns:
            Chunks belonging to the document, sorted by index.
        """
        all_memories = await self.list(limit=10000)

        chunks = [
            m for m in all_memories
            if m.metadata.get("document_id") == document_id
        ]

        # Sort by chunk index
        chunks.sort(key=lambda m: m.metadata.get("chunk_index", 0))
        return chunks

    async def update_embedding(self, memory_id: MemoryId) -> bool:
        """Regenerate the embedding for a memory.

        Args:
            memory_id: The ID of the memory to update.

        Returns:
            True if updated, False if not found.
        """
        if self._embedding_provider is None:
            return False

        memory = await self._backend.retrieve(memory_id)
        if memory is None:
            return False

        new_embedding = await self._embedding_provider.embed(memory.content)
        updated = memory.with_embedding(new_embedding)

        return await self._backend.update(updated)

    async def reindex_all(self) -> int:
        """Regenerate embeddings for all memories.

        Returns:
            Number of memories reindexed.
        """
        if self._embedding_provider is None:
            return 0

        all_memories = await self.list(limit=100000)
        count = 0

        for memory in all_memories:
            if await self.update_embedding(memory.id):
                count += 1

        return count
