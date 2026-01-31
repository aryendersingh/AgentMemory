"""AI Agent Memory Framework - A developer-friendly Python API for AI agent memory.

This framework provides pluggable storage backends and multiple memory types
designed for autonomous AI agents.

Example:
    from agent_memory import AgentMemory, MemoryConfig

    memory = AgentMemory(
        agent_id="my-agent",
        config=MemoryConfig(
            backend="memory",
            embedding_provider="dummy",
        )
    )

    # Short-term memory
    await memory.short_term.add("Current task context")

    # Long-term memory
    await memory.long_term.store_fact("User's name is Alice")

    # Conversation memory
    await memory.conversation.add_message(role="user", content="Hello!")

    # Cleanup
    await memory.close()
"""

from agent_memory.agent import AgentMemory
from agent_memory.backends.memory_backend import InMemoryBackend
from agent_memory.config.settings import (
    BackendType,
    EmbeddingProviderType,
    MemoryConfig,
)
from agent_memory.core.exceptions import (
    AgentMemoryError,
    BackendError,
    EmbeddingError,
    MemoryNotFoundError,
    ValidationError,
)
from agent_memory.core.memory import MemoryQuery, MemoryRecord, ScoredMemory
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.embeddings.base import DummyEmbeddingProvider
from agent_memory.memories.conversation import ConversationMemory
from agent_memory.memories.episodic import EpisodicMemory
from agent_memory.memories.long_term import LongTermMemory
from agent_memory.memories.procedural import ProceduralMemory
from agent_memory.memories.semantic import SemanticMemory
from agent_memory.memories.short_term import ShortTermMemory

__version__ = "0.1.0"

__all__ = [
    # Main facade
    "AgentMemory",
    # Configuration
    "MemoryConfig",
    "BackendType",
    "EmbeddingProviderType",
    # Core types
    "MemoryId",
    "MemoryType",
    "MemoryRecord",
    "MemoryQuery",
    "ScoredMemory",
    # Protocols
    "StorageBackend",
    "EmbeddingProvider",
    # Memory types
    "ShortTermMemory",
    "LongTermMemory",
    "ConversationMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    # Backends
    "InMemoryBackend",
    # Embedding providers
    "DummyEmbeddingProvider",
    # Exceptions
    "AgentMemoryError",
    "BackendError",
    "EmbeddingError",
    "MemoryNotFoundError",
    "ValidationError",
]
