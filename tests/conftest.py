"""Pytest configuration and fixtures for agent memory tests."""

import pytest

from agent_memory import AgentMemory, MemoryConfig
from agent_memory.backends.memory_backend import InMemoryBackend
from agent_memory.embeddings.base import DummyEmbeddingProvider


@pytest.fixture
def backend():
    """Provide a fresh in-memory backend for each test."""
    return InMemoryBackend()


@pytest.fixture
def embedding_provider():
    """Provide a dummy embedding provider for tests."""
    return DummyEmbeddingProvider(dimension=384)


@pytest.fixture
async def memory(backend, embedding_provider):
    """Provide a configured AgentMemory instance."""
    mem = AgentMemory(
        agent_id="test-agent",
        config=MemoryConfig(
            backend="memory",
            embedding_provider="dummy",
        ),
        backend=backend,
        embedding_provider=embedding_provider,
    )
    yield mem
    await mem.close()


@pytest.fixture
def config():
    """Provide a default memory configuration."""
    return MemoryConfig(
        backend="memory",
        embedding_provider="dummy",
    )
