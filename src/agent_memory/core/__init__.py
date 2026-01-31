"""Core types and protocols for the agent memory framework."""

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

__all__ = [
    "MemoryId",
    "MemoryType",
    "MemoryRecord",
    "MemoryQuery",
    "ScoredMemory",
    "StorageBackend",
    "EmbeddingProvider",
    "AgentMemoryError",
    "BackendError",
    "EmbeddingError",
    "MemoryNotFoundError",
    "ValidationError",
]
