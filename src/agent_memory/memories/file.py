"""File memory for indexing and searching file contents."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_memory.core.memory import MemoryRecord, ScoredMemory
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.files.loader import FileLoader
from agent_memory.files.types import ChunkingConfig, ExtractedFile, FileMetadata
from agent_memory.memories.base import BaseMemory


class FileMemory(BaseMemory):
    """Memory for indexing and searching file contents.

    FileMemory allows you to add files to the memory system where they
    are read, chunked, embedded, and stored for semantic search.

    Requires an embedding provider to be configured.

    Example:
        memory = AgentMemory(agent_id="my-agent", config=config)
        file_id = await memory.file.add_file("./docs/readme.md")
        results = await memory.file.search_files("configuration")
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider,
        chunking_config: ChunkingConfig | None = None,
        file_loader: FileLoader | None = None,
    ) -> None:
        """Initialize file memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: The embedding provider (required).
            chunking_config: Configuration for chunking files.
            file_loader: Custom file loader. Uses default if not provided.
        """
        super().__init__(backend, agent_id, embedding_provider)
        if embedding_provider is None:
            raise ValueError("FileMemory requires an embedding provider")

        self._chunking_config = chunking_config or ChunkingConfig()
        self._file_loader = file_loader or FileLoader(chunking_config=self._chunking_config)

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.FILE

    async def add_file(
        self,
        path: str | Path,
        importance: float = 0.5,
        tags: list[str] | None = None,
        chunking_config: ChunkingConfig | None = None,
        skip_if_exists: bool = True,
    ) -> MemoryId:
        """Add a file to memory.

        The file is read, chunked, and each chunk is stored with
        an embedding for semantic search.

        Args:
            path: Path to the file to add.
            importance: Importance score for the file chunks.
            tags: Optional tags for categorization.
            chunking_config: Override chunking configuration.
            skip_if_exists: If True, skip if file with same hash exists.

        Returns:
            The ID of the file record (first chunk or main record).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        path = Path(path).resolve()

        # Load and extract file content
        extracted, chunks = self._file_loader.load_and_chunk(
            path,
            config=chunking_config,
        )

        # Check if file already exists with same hash
        if skip_if_exists:
            existing = await self.get_by_hash(extracted.metadata.content_hash)
            if existing:
                return existing[0].id

        # Build base metadata
        base_metadata: dict[str, Any] = {
            **extracted.metadata.to_dict(),
            "is_file": True,
        }
        if tags:
            base_metadata["tags"] = tags
        if extracted.extra:
            base_metadata.update(extracted.extra)

        # Store each chunk
        chunk_ids: list[MemoryId] = []
        for chunk in chunks:
            chunk_metadata = {
                **base_metadata,
                **chunk.to_metadata(),
            }

            record = await self._create_record(
                content=chunk.content,
                importance=importance,
                metadata=chunk_metadata,
                ttl=None,
                generate_embedding=True,
            )
            memory_id = await self._backend.store(record)
            chunk_ids.append(memory_id)

        # Return the first chunk's ID as the file ID
        return chunk_ids[0]

    async def add_file_content(
        self,
        content: str,
        metadata: FileMetadata,
        importance: float = 0.5,
        tags: list[str] | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> MemoryId:
        """Add pre-extracted file content to memory.

        Use this when you've already extracted the content from a file.

        Args:
            content: The text content of the file.
            metadata: File metadata.
            importance: Importance score.
            tags: Optional tags.
            chunking_config: Override chunking configuration.

        Returns:
            The ID of the file record (first chunk).
        """
        # Chunk the content
        chunks = self._file_loader.chunk(
            content,
            config=chunking_config,
        )

        # Build base metadata
        base_metadata: dict[str, Any] = {
            **metadata.to_dict(),
            "is_file": True,
        }
        if tags:
            base_metadata["tags"] = tags

        # Store each chunk
        chunk_ids: list[MemoryId] = []
        for chunk in chunks:
            chunk_metadata = {
                **base_metadata,
                **chunk.to_metadata(),
            }

            record = await self._create_record(
                content=chunk.content,
                importance=importance,
                metadata=chunk_metadata,
                ttl=None,
                generate_embedding=True,
            )
            memory_id = await self._backend.store(record)
            chunk_ids.append(memory_id)

        return chunk_ids[0]

    async def search_files(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        tags: list[str] | None = None,
        file_extensions: list[str] | None = None,
    ) -> Sequence[ScoredMemory]:
        """Search file contents semantically.

        Args:
            query: The text query to search for.
            limit: Maximum number of results.
            min_score: Minimum similarity score.
            tags: Optional filter by tags.
            file_extensions: Optional filter by file extensions.

        Returns:
            Sequence of scored memories, sorted by similarity.
        """
        if self._embedding_provider is None:
            raise ValueError("No embedding provider configured")

        embedding = await self._embedding_provider.embed(query)
        results = await self._backend.search_similar(
            embedding=embedding,
            limit=limit * 3,  # Get more for filtering
            min_score=min_score,
            memory_type=self.memory_type.value,
            agent_id=self._agent_id,
        )

        # Filter results
        filtered: list[ScoredMemory] = []
        for scored in results:
            memory = scored.memory
            meta = memory.metadata

            # Must be a file memory
            if not meta.get("is_file"):
                continue

            # Apply tag filter
            if tags:
                memory_tags = meta.get("tags", [])
                if not any(tag in memory_tags for tag in tags):
                    continue

            # Apply extension filter
            if file_extensions:
                ext = meta.get("file_extension", "")
                if ext not in file_extensions:
                    continue

            filtered.append(scored)
            if len(filtered) >= limit:
                break

        return filtered

    async def search_by_filename(
        self,
        pattern: str,
        limit: int = 100,
    ) -> Sequence[MemoryRecord]:
        """Search for files by filename pattern.

        Args:
            pattern: Glob-style pattern (e.g., "*.md", "readme*").
            limit: Maximum number of results.

        Returns:
            Sequence of matching file records.
        """
        # Get all file memories
        all_files = await self.list_files(limit=10000)

        # Filter by pattern
        matched: list[MemoryRecord] = []
        seen_paths: set[str] = set()

        for memory in all_files:
            file_name = memory.metadata.get("file_name", "")
            file_path = memory.metadata.get("file_path", "")

            # Skip if we've already seen this file
            if file_path in seen_paths:
                continue

            if fnmatch.fnmatch(file_name.lower(), pattern.lower()):
                matched.append(memory)
                seen_paths.add(file_path)

                if len(matched) >= limit:
                    break

        return matched

    async def list_files(
        self,
        limit: int = 100,
        tags: list[str] | None = None,
    ) -> Sequence[MemoryRecord]:
        """List all indexed files.

        Returns first chunk of each unique file.

        Args:
            limit: Maximum number of files to return.
            tags: Optional filter by tags.

        Returns:
            Sequence of file records (first chunk of each file).
        """
        all_memories = await self.list(limit=limit * 10, tags=tags)

        # Group by file path and get first chunk of each
        seen_paths: set[str] = set()
        files: list[MemoryRecord] = []

        for memory in all_memories:
            if not memory.metadata.get("is_file"):
                continue

            file_path = memory.metadata.get("file_path", "")
            if file_path in seen_paths:
                continue

            # Only include chunk 0 (first chunk)
            if memory.metadata.get("chunk_index", 0) == 0:
                seen_paths.add(file_path)
                files.append(memory)

                if len(files) >= limit:
                    break

        return files

    async def get_file_chunks(
        self,
        file_path: str | Path,
    ) -> Sequence[MemoryRecord]:
        """Get all chunks for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            Sequence of chunk records, sorted by index.
        """
        file_path = str(Path(file_path).resolve())
        all_memories = await self.list(limit=10000)

        chunks = [
            m for m in all_memories
            if m.metadata.get("file_path") == file_path
        ]

        # Sort by chunk index
        chunks.sort(key=lambda m: m.metadata.get("chunk_index", 0))
        return chunks

    async def get_by_hash(
        self,
        content_hash: str,
    ) -> Sequence[MemoryRecord]:
        """Get file records by content hash.

        Args:
            content_hash: SHA-256 hash of the file content.

        Returns:
            Sequence of matching records.
        """
        all_memories = await self.list(limit=10000)

        return [
            m for m in all_memories
            if m.metadata.get("file_hash") == content_hash
        ]

    async def update_file(
        self,
        path: str | Path,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryId | None:
        """Re-index a file if it has changed.

        Compares the current file hash with the stored hash.
        If different, removes old chunks and adds new ones.

        Args:
            path: Path to the file.
            importance: Importance score for new chunks.
            tags: Tags for the new chunks.

        Returns:
            New file ID if updated, None if unchanged.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Calculate current hash
        content_bytes = path.read_bytes()
        import hashlib
        current_hash = hashlib.sha256(content_bytes).hexdigest()

        # Check existing file
        existing = await self.get_file_chunks(path)
        if existing:
            existing_hash = existing[0].metadata.get("file_hash")
            if existing_hash == current_hash:
                # File unchanged
                return None

            # Remove old chunks
            await self.remove_by_path(path)

        # Add updated file
        return await self.add_file(
            path,
            importance=importance,
            tags=tags,
            skip_if_exists=False,
        )

    async def remove_by_path(
        self,
        path: str | Path,
    ) -> int:
        """Remove all chunks for a file by path.

        Args:
            path: Path to the file.

        Returns:
            Number of chunks removed.
        """
        chunks = await self.get_file_chunks(path)
        count = 0

        for chunk in chunks:
            if await self._backend.delete(chunk.id):
                count += 1

        return count

    async def get_file_content(
        self,
        file_path: str | Path,
    ) -> str | None:
        """Reconstruct file content from stored chunks.

        Args:
            file_path: Path to the file.

        Returns:
            Reconstructed content, or None if file not found.
        """
        chunks = await self.get_file_chunks(file_path)
        if not chunks:
            return None

        # Sort by chunk index and concatenate
        # Note: Due to overlapping, we need to be careful about reconstruction
        # For now, just concatenate the chunks (may have some overlap issues)
        return "".join(chunk.content for chunk in chunks)

    async def get_file_metadata(
        self,
        file_path: str | Path,
    ) -> dict[str, Any] | None:
        """Get metadata for a file.

        Args:
            file_path: Path to the file.

        Returns:
            File metadata dictionary, or None if not found.
        """
        chunks = await self.get_file_chunks(file_path)
        if not chunks:
            return None

        # Return metadata from first chunk
        meta = dict(chunks[0].metadata)
        # Remove chunk-specific fields
        for key in ["chunk_index", "chunk_start", "chunk_end", "total_chunks"]:
            meta.pop(key, None)

        return meta
