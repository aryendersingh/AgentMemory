"""AgentMemory facade class providing unified access to all memory types."""

from __future__ import annotations

from typing import Any

from agent_memory.backends.memory_backend import InMemoryBackend
from agent_memory.config.settings import (
    BackendType,
    EmbeddingProviderType,
    MemoryConfig,
)
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.embeddings.base import DummyEmbeddingProvider
from agent_memory.memories.conversation import ConversationMemory
from agent_memory.memories.episodic import EpisodicMemory
from agent_memory.memories.file import FileMemory
from agent_memory.memories.long_term import LongTermMemory
from agent_memory.memories.procedural import ProceduralMemory
from agent_memory.memories.semantic import SemanticMemory
from agent_memory.memories.short_term import ShortTermMemory


class AgentMemory:
    """Main facade for the agent memory system.

    Provides unified access to all memory types through a single interface.

    Example:
        memory = AgentMemory(agent_id="my-agent")
        await memory.short_term.add("Current task context")
        await memory.long_term.store_fact("User prefers dark mode")
        await memory.close()
    """

    def __init__(
        self,
        agent_id: str = "default",
        config: MemoryConfig | None = None,
        backend: StorageBackend | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize the agent memory system.

        Args:
            agent_id: Unique identifier for this agent.
            config: Configuration settings. Uses defaults if not provided.
            backend: Custom storage backend. If not provided, uses config.backend.
            embedding_provider: Custom embedding provider. If not provided, uses config.
        """
        self._agent_id = agent_id
        self._config = config or MemoryConfig()
        self._backend = backend or self._create_backend()
        self._embedding_provider = embedding_provider or self._create_embedding_provider()

        # Initialize memory types
        self._short_term = ShortTermMemory(
            backend=self._backend,
            agent_id=self._agent_id,
            embedding_provider=self._embedding_provider,
            default_ttl=self._config.short_term_ttl,
        )

        self._long_term = LongTermMemory(
            backend=self._backend,
            agent_id=self._agent_id,
            embedding_provider=self._embedding_provider,
        )

        self._conversation = ConversationMemory(
            backend=self._backend,
            agent_id=self._agent_id,
            embedding_provider=self._embedding_provider,
            max_messages=self._config.conversation_max_messages,
        )

        self._episodic = EpisodicMemory(
            backend=self._backend,
            agent_id=self._agent_id,
            embedding_provider=self._embedding_provider,
        )

        self._procedural = ProceduralMemory(
            backend=self._backend,
            agent_id=self._agent_id,
            embedding_provider=self._embedding_provider,
        )

        self._semantic: SemanticMemory | None = None
        if self._embedding_provider is not None:
            self._semantic = SemanticMemory(
                backend=self._backend,
                agent_id=self._agent_id,
                embedding_provider=self._embedding_provider,
            )

        self._file: FileMemory | None = None
        if self._embedding_provider is not None:
            self._file = FileMemory(
                backend=self._backend,
                agent_id=self._agent_id,
                embedding_provider=self._embedding_provider,
            )

    def _create_backend(self) -> StorageBackend:
        """Create the storage backend based on configuration."""
        backend_type = self._config.backend

        if backend_type == BackendType.MEMORY:
            return InMemoryBackend()

        if backend_type == BackendType.REDIS:
            try:
                from agent_memory.backends.redis_backend import RedisBackend

                return RedisBackend(**self._config.backend_config)
            except ImportError:
                raise ImportError(
                    "Redis backend requires redis package. "
                    "Install with: pip install agent-memory[redis]"
                )

        if backend_type == BackendType.POSTGRES:
            try:
                from agent_memory.backends.postgres_backend import PostgresBackend

                return PostgresBackend(**self._config.backend_config)
            except ImportError:
                raise ImportError(
                    "PostgreSQL backend requires asyncpg package. "
                    "Install with: pip install agent-memory[postgres]"
                )

        if backend_type == BackendType.CHROMA:
            try:
                from agent_memory.backends.vector.chroma_backend import ChromaBackend

                return ChromaBackend(**self._config.backend_config)
            except ImportError:
                raise ImportError(
                    "ChromaDB backend requires chromadb package. "
                    "Install with: pip install agent-memory[chroma]"
                )

        # Default to in-memory
        return InMemoryBackend()

    def _create_embedding_provider(self) -> EmbeddingProvider | None:
        """Create the embedding provider based on configuration."""
        provider_type = self._config.embedding_provider

        if provider_type == EmbeddingProviderType.NONE:
            return None

        if provider_type == EmbeddingProviderType.DUMMY:
            return DummyEmbeddingProvider(
                dimension=self._config.embedding_config.get("dimension", 384)
            )

        if provider_type == EmbeddingProviderType.OPENAI:
            try:
                from agent_memory.embeddings.openai_embeddings import (
                    OpenAIEmbeddingProvider,
                )

                return OpenAIEmbeddingProvider(**self._config.embedding_config)
            except ImportError:
                raise ImportError(
                    "OpenAI embeddings require openai package. "
                    "Install with: pip install agent-memory[openai]"
                )

        return None

    @property
    def agent_id(self) -> str:
        """The ID of this agent."""
        return self._agent_id

    @property
    def short_term(self) -> ShortTermMemory:
        """Access short-term memory for current session/task context."""
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        """Access long-term memory for persistent facts and preferences."""
        return self._long_term

    @property
    def conversation(self) -> ConversationMemory:
        """Access conversation memory for chat history."""
        return self._conversation

    @property
    def episodic(self) -> EpisodicMemory:
        """Access episodic memory for events and experiences."""
        return self._episodic

    @property
    def procedural(self) -> ProceduralMemory:
        """Access procedural memory for skills and workflows."""
        return self._procedural

    @property
    def semantic(self) -> SemanticMemory:
        """Access semantic memory for vector-based search.

        Raises:
            ValueError: If no embedding provider is configured.
        """
        if self._semantic is None:
            raise ValueError(
                "Semantic memory requires an embedding provider. "
                "Configure with embedding_provider='openai' or 'dummy'."
            )
        return self._semantic

    @property
    def has_semantic(self) -> bool:
        """Whether semantic memory is available."""
        return self._semantic is not None

    @property
    def file(self) -> FileMemory:
        """Access file memory for indexing and searching file contents.

        Raises:
            ValueError: If no embedding provider is configured.
        """
        if self._file is None:
            raise ValueError(
                "File memory requires an embedding provider. "
                "Configure with embedding_provider='openai' or 'dummy'."
            )
        return self._file

    @property
    def has_file(self) -> bool:
        """Whether file memory is available."""
        return self._file is not None

    @property
    def backend(self) -> StorageBackend:
        """The underlying storage backend."""
        return self._backend

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The embedding provider, if configured."""
        return self._embedding_provider

    async def clear_all(self) -> dict[str, int]:
        """Clear all memories for this agent.

        Returns:
            Dictionary with counts of cleared memories per type.
        """
        counts: dict[str, int] = {}

        counts["short_term"] = await self._short_term.clear()
        counts["long_term"] = await self._long_term.clear()
        counts["conversation"] = await self._conversation.clear()
        counts["episodic"] = await self._episodic.clear()
        counts["procedural"] = await self._procedural.clear()

        if self._semantic is not None:
            counts["semantic"] = await self._semantic.clear()

        if self._file is not None:
            counts["file"] = await self._file.clear()

        return counts

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored memories.

        Returns:
            Dictionary with memory counts and statistics.
        """
        stats: dict[str, Any] = {
            "agent_id": self._agent_id,
            "short_term": await self._short_term.count(),
            "long_term": await self._long_term.count(),
            "conversation": await self._conversation.count(),
            "episodic": await self._episodic.count(),
            "procedural": await self._procedural.count(),
        }

        if self._semantic is not None:
            stats["semantic"] = await self._semantic.count()

        if self._file is not None:
            stats["file"] = await self._file.count()

        if hasattr(self._backend, "get_stats"):
            stats["backend"] = await self._backend.get_stats()

        return stats

    async def close(self) -> None:
        """Close the memory system and release resources."""
        await self._backend.close()
        if self._embedding_provider is not None:
            await self._embedding_provider.close()

    async def __aenter__(self) -> "AgentMemory":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
