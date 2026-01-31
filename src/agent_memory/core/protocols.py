"""Protocol definitions for pluggable backends and providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from agent_memory.core.memory import MemoryQuery, MemoryRecord, ScoredMemory
    from agent_memory.core.types import MemoryId



@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for memory storage backends.

    Implementations must provide async methods for CRUD operations
    and querying memories. This enables pluggable storage solutions
    like in-memory, Redis, PostgreSQL, or vector databases.
    """

    async def store(self, record: MemoryRecord) -> MemoryId:
        """Store a memory record and return its ID.

        Args:
            record: The memory record to store.

        Returns:
            The ID of the stored record.

        Raises:
            BackendError: If storage fails.
        """
        ...

    async def retrieve(self, memory_id: MemoryId) -> MemoryRecord | None:
        """Retrieve a memory record by its ID.

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The memory record if found, None otherwise.
        """
        ...

    async def update(self, record: MemoryRecord) -> bool:
        """Update an existing memory record.

        Args:
            record: The updated memory record (must have existing ID).

        Returns:
            True if the record was updated, False if not found.
        """
        ...

    async def delete(self, memory_id: MemoryId) -> bool:
        """Delete a memory record.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    async def query(self, query: MemoryQuery) -> Sequence[MemoryRecord]:
        """Query memories based on filter criteria.

        Args:
            query: Query parameters for filtering memories.

        Returns:
            Sequence of matching memory records.
        """
        ...

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        memory_type: str | None = None,
        agent_id: str | None = None,
    ) -> Sequence[ScoredMemory]:
        """Search for similar memories using vector similarity.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results.
            min_score: Minimum similarity score threshold.
            memory_type: Optional filter by memory type.
            agent_id: Optional filter by agent ID.

        Returns:
            Sequence of scored memories, sorted by similarity.
        """
        ...

    async def clear(self, agent_id: str | None = None) -> int:
        """Clear memories, optionally filtered by agent ID.

        Args:
            agent_id: If provided, only clear memories for this agent.

        Returns:
            Number of memories deleted.
        """
        ...

    async def close(self) -> None:
        """Close the backend and release resources."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implementations convert text into vector embeddings
    for semantic search and similarity matching.
    """

    @property
    def dimension(self) -> int:
        """The dimension of embeddings produced by this provider."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        ...

    async def close(self) -> None:
        """Close the provider and release resources."""
        ...
