"""Configuration dataclasses for the agent memory framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any


class BackendType(StrEnum):
    """Supported storage backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    POSTGRES = "postgres"
    CHROMA = "chroma"
    PINECONE = "pinecone"


class EmbeddingProviderType(StrEnum):
    """Supported embedding provider types."""

    NONE = "none"
    DUMMY = "dummy"
    OPENAI = "openai"


@dataclass
class MemoryConfig:
    """Configuration for the agent memory system.

    Attributes:
        backend: The storage backend to use.
        embedding_provider: The embedding provider for semantic search.
        backend_config: Backend-specific configuration.
        embedding_config: Embedding provider-specific configuration.
        short_term_ttl: Default TTL for short-term memories.
        conversation_max_messages: Maximum messages per conversation.
        enable_scoring: Whether to enable importance scoring.
        enable_decay: Whether to enable memory decay.
        decay_half_life: Half-life for exponential decay.
    """

    backend: BackendType | str = BackendType.MEMORY
    embedding_provider: EmbeddingProviderType | str = EmbeddingProviderType.NONE
    backend_config: dict[str, Any] = field(default_factory=dict)
    embedding_config: dict[str, Any] = field(default_factory=dict)

    # Short-term memory settings
    short_term_ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))

    # Conversation memory settings
    conversation_max_messages: int = 100

    # Scoring settings
    enable_scoring: bool = True
    enable_decay: bool = True
    decay_half_life: timedelta = field(default_factory=lambda: timedelta(days=7))

    def __post_init__(self) -> None:
        """Normalize string values to enums."""
        if isinstance(self.backend, str):
            try:
                self.backend = BackendType(self.backend)
            except ValueError:
                pass  # Keep as string for custom backends

        if isinstance(self.embedding_provider, str):
            try:
                self.embedding_provider = EmbeddingProviderType(self.embedding_provider)
            except ValueError:
                pass  # Keep as string for custom providers


@dataclass
class RedisConfig:
    """Configuration for Redis backend."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    key_prefix: str = "agent_memory:"
    ssl: bool = False


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL backend."""

    host: str = "localhost"
    port: int = 5432
    database: str = "agent_memory"
    user: str = "postgres"
    password: str | None = None
    table_name: str = "memories"
    enable_vector: bool = True


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB backend."""

    collection_name: str = "agent_memory"
    persist_directory: str | None = None
    host: str | None = None
    port: int | None = None


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for OpenAI embeddings."""

    api_key: str | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
