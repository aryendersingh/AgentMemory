"""Embedding providers for semantic memory."""

from agent_memory.embeddings.base import DummyEmbeddingProvider

__all__ = ["DummyEmbeddingProvider"]

# Optional imports for installed providers
try:
    from agent_memory.embeddings.openai_embeddings import OpenAIEmbeddingProvider

    __all__.append("OpenAIEmbeddingProvider")
except ImportError:
    pass
