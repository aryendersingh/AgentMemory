"""OpenAI embedding provider implementation."""

from __future__ import annotations

from typing import Any

from agent_memory.core.exceptions import EmbeddingError

# Model dimensions for OpenAI embedding models
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider:
    """Embedding provider using OpenAI's embedding API.

    Requires the openai package to be installed:
        pip install agent-memory[openai]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            model: The embedding model to use.
            dimensions: Optional dimension override (for models that support it).
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install agent-memory[openai]"
            )

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions or MODEL_DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        """The dimension of embeddings produced by this provider."""
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for text using OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            EmbeddingError: If the API call fails.
        """
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(str(e), provider="openai") from e

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as e:
            raise EmbeddingError(str(e), provider="openai") from e

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()
