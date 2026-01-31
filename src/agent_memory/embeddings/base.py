"""Base embedding provider implementations."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:
        """The dimension of embeddings produced by this provider."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    async def close(self) -> None:
        """Close the provider and release resources."""
        ...


class DummyEmbeddingProvider:
    """A deterministic dummy embedding provider for testing.

    Generates consistent embeddings based on text content hashing.
    Not suitable for production use - use for testing only.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize the dummy provider.

        Args:
            dimension: The dimension of embeddings to generate.
        """
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The dimension of embeddings produced by this provider."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding for text.

        Uses SHA-256 hash to generate consistent embeddings.

        Args:
            text: The text to embed.

        Returns:
            A normalized embedding vector.
        """
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Generate values from hash
        values: list[float] = []
        for i in range(self._dimension):
            # Use hash bytes cyclically
            byte_idx = i % len(hash_bytes)
            # Convert to float in range [-1, 1]
            values.append((hash_bytes[byte_idx] / 127.5) - 1.0)

        # Normalize the vector
        return self._normalize(values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return [await self.embed(text) for text in texts]

    def _normalize(self, vector: list[float]) -> list[float]:
        """Normalize a vector to unit length.

        Args:
            vector: The vector to normalize.

        Returns:
            Normalized vector.
        """
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return vector
        return [x / magnitude for x in vector]

    async def close(self) -> None:
        """Close the provider (no-op for dummy provider)."""
        pass


class CachedEmbeddingProvider:
    """Wrapper that caches embeddings to avoid redundant API calls."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        max_cache_size: int = 10000,
    ) -> None:
        """Initialize the cached provider.

        Args:
            provider: The underlying embedding provider.
            max_cache_size: Maximum number of embeddings to cache.
        """
        self._provider = provider
        self._cache: dict[str, list[float]] = {}
        self._max_size = max_cache_size

    @property
    def dimension(self) -> int:
        """The dimension of embeddings produced by this provider."""
        return self._provider.dimension

    async def embed(self, text: str) -> list[float]:
        """Generate or retrieve cached embedding for text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        cache_key = self._hash_text(text)

        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = await self._provider.embed(text)

        # Add to cache, evicting oldest if necessary
        if len(self._cache) >= self._max_size:
            # Remove first item (oldest)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        results: list[list[float]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._hash_text(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append([])  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Batch embed uncached texts
        if uncached_texts:
            new_embeddings = await self._provider.embed_batch(uncached_texts)

            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                cache_key = self._hash_text(texts[idx])

                if len(self._cache) >= self._max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

                self._cache[cache_key] = embedding

        return results

    def _hash_text(self, text: str) -> str:
        """Generate a cache key for text.

        Args:
            text: The text to hash.

        Returns:
            A hash string for use as cache key.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    async def close(self) -> None:
        """Close the underlying provider."""
        await self._provider.close()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Current number of cached embeddings."""
        return len(self._cache)
