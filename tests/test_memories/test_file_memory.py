"""Tests for FileMemory class."""

from pathlib import Path

import pytest

from agent_memory import AgentMemory, MemoryConfig
from agent_memory.backends.memory_backend import InMemoryBackend
from agent_memory.embeddings.base import DummyEmbeddingProvider
from agent_memory.files.types import ChunkingConfig
from agent_memory.memories.file import FileMemory


@pytest.fixture
def backend() -> InMemoryBackend:
    """Provide a fresh in-memory backend."""
    return InMemoryBackend()


@pytest.fixture
def embedding_provider() -> DummyEmbeddingProvider:
    """Provide a dummy embedding provider."""
    return DummyEmbeddingProvider(dimension=384)


@pytest.fixture
def file_memory(backend: InMemoryBackend, embedding_provider: DummyEmbeddingProvider) -> FileMemory:
    """Provide a FileMemory instance."""
    return FileMemory(
        backend=backend,
        agent_id="test-agent",
        embedding_provider=embedding_provider,
    )


@pytest.fixture
async def memory_with_file(backend: InMemoryBackend, embedding_provider: DummyEmbeddingProvider):
    """Provide an AgentMemory instance with file support."""
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
def fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path(__file__).parent.parent / "test_files" / "fixtures"


class TestFileMemoryInit:
    """Tests for FileMemory initialization."""

    def test_requires_embedding_provider(self, backend: InMemoryBackend) -> None:
        """Test that FileMemory requires an embedding provider."""
        with pytest.raises(ValueError, match="requires an embedding provider"):
            FileMemory(
                backend=backend,
                agent_id="test",
                embedding_provider=None,  # type: ignore
            )

    def test_file_memory_type(self, file_memory: FileMemory) -> None:
        """Test that memory type is FILE."""
        from agent_memory.core.types import MemoryType
        assert file_memory.memory_type == MemoryType.FILE


class TestAddFile:
    """Tests for adding files to memory."""

    async def test_add_text_file(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test adding a text file."""
        file_id = await file_memory.add_file(fixtures_dir / "sample.txt")

        assert file_id is not None
        count = await file_memory.count()
        assert count >= 1

    async def test_add_file_with_tags(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test adding a file with tags."""
        file_id = await file_memory.add_file(
            fixtures_dir / "sample.txt",
            tags=["test", "documentation"],
        )

        chunks = await file_memory.get_file_chunks(fixtures_dir / "sample.txt")
        assert len(chunks) >= 1
        assert "test" in chunks[0].metadata.get("tags", [])

    async def test_add_file_stores_metadata(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test that file metadata is stored correctly."""
        await file_memory.add_file(fixtures_dir / "sample.txt")

        chunks = await file_memory.get_file_chunks(fixtures_dir / "sample.txt")
        assert len(chunks) >= 1

        metadata = chunks[0].metadata
        assert metadata.get("file_name") == "sample.txt"
        assert metadata.get("file_extension") == "txt"
        assert metadata.get("file_hash") is not None
        assert metadata.get("is_file") is True

    async def test_add_file_not_found(self, file_memory: FileMemory) -> None:
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            await file_memory.add_file("/nonexistent/file.txt")


class TestHashBasedDeduplication:
    """Tests for hash-based deduplication."""

    async def test_skip_duplicate_file(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test that duplicate files are skipped by default."""
        file_id1 = await file_memory.add_file(fixtures_dir / "sample.txt")
        file_id2 = await file_memory.add_file(fixtures_dir / "sample.txt")

        # Should return the same ID (skipped duplicate)
        assert file_id1 == file_id2

    async def test_force_add_duplicate(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test adding duplicate file when skip_if_exists=False."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        count1 = await file_memory.count()

        await file_memory.add_file(fixtures_dir / "sample.txt", skip_if_exists=False)
        count2 = await file_memory.count()

        # Should have more chunks after adding again
        assert count2 > count1


class TestSearchFiles:
    """Tests for searching file contents."""

    async def test_search_files_basic(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test basic semantic search returns results (may be empty with dummy embeddings)."""
        await file_memory.add_file(fixtures_dir / "sample.txt")

        # With dummy embeddings, similarity scores may be low
        # Just verify the search runs without error and respects filters
        results = await file_memory.search_files("sample text file", min_score=-1.0)

        # Results should only include file memories
        for scored in results:
            assert scored.memory.metadata.get("is_file") is True

    async def test_search_with_tags(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test search with tag filter."""
        await file_memory.add_file(fixtures_dir / "sample.txt", tags=["docs"])
        await file_memory.add_file(fixtures_dir / "sample.py", tags=["code"])

        # Search with tag filter
        results = await file_memory.search_files("content", tags=["docs"])

        # Results should only include tagged files
        for scored in results:
            tags = scored.memory.metadata.get("tags", [])
            assert "docs" in tags

    async def test_search_with_extension_filter(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test search with file extension filter."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        await file_memory.add_file(fixtures_dir / "sample.py")

        results = await file_memory.search_files(
            "content",
            file_extensions=["py"],
        )

        for scored in results:
            assert scored.memory.metadata.get("file_extension") == "py"


class TestSearchByFilename:
    """Tests for filename pattern search."""

    async def test_search_by_exact_name(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test searching by exact filename."""
        await file_memory.add_file(fixtures_dir / "sample.txt")

        results = await file_memory.search_by_filename("sample.txt")

        assert len(results) == 1
        assert results[0].metadata.get("file_name") == "sample.txt"

    async def test_search_by_pattern(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test searching by glob pattern."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        await file_memory.add_file(fixtures_dir / "sample.md")
        await file_memory.add_file(fixtures_dir / "sample.py")

        results = await file_memory.search_by_filename("sample.*")

        assert len(results) == 3

    async def test_search_by_extension_pattern(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test searching by extension pattern."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        await file_memory.add_file(fixtures_dir / "sample.py")

        results = await file_memory.search_by_filename("*.py")

        assert len(results) == 1
        assert results[0].metadata.get("file_extension") == "py"


class TestListFiles:
    """Tests for listing indexed files."""

    async def test_list_files(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test listing all indexed files."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        await file_memory.add_file(fixtures_dir / "sample.md")

        files = await file_memory.list_files()

        assert len(files) == 2
        names = {f.metadata.get("file_name") for f in files}
        assert "sample.txt" in names
        assert "sample.md" in names

    async def test_list_files_with_tags(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test listing files with tag filter."""
        await file_memory.add_file(fixtures_dir / "sample.txt", tags=["docs"])
        await file_memory.add_file(fixtures_dir / "sample.py", tags=["code"])

        files = await file_memory.list_files(tags=["docs"])

        assert len(files) == 1
        assert files[0].metadata.get("file_name") == "sample.txt"


class TestGetFileChunks:
    """Tests for getting file chunks."""

    async def test_get_chunks(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test getting all chunks for a file."""
        await file_memory.add_file(fixtures_dir / "sample.txt")

        chunks = await file_memory.get_file_chunks(fixtures_dir / "sample.txt")

        assert len(chunks) >= 1
        # Chunks should be sorted by index
        indices = [c.metadata.get("chunk_index", 0) for c in chunks]
        assert indices == sorted(indices)

    async def test_get_chunks_nonexistent(self, file_memory: FileMemory) -> None:
        """Test getting chunks for non-indexed file."""
        chunks = await file_memory.get_file_chunks("/nonexistent/file.txt")

        assert len(chunks) == 0


class TestUpdateFile:
    """Tests for updating files."""

    async def test_update_unchanged_file(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test that unchanged file is not re-indexed."""
        await file_memory.add_file(fixtures_dir / "sample.txt")

        result = await file_memory.update_file(fixtures_dir / "sample.txt")

        # Should return None (no update needed)
        assert result is None

    async def test_update_changed_file(self, file_memory: FileMemory, tmp_path: Path) -> None:
        """Test that changed file is re-indexed."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")

        await file_memory.add_file(test_file)
        count1 = await file_memory.count()

        # Modify file
        test_file.write_text("Modified content")

        result = await file_memory.update_file(test_file)

        assert result is not None
        count2 = await file_memory.count()
        # Count should be same (old chunks removed, new added)
        assert count2 == count1

    async def test_update_nonexistent_file(self, file_memory: FileMemory) -> None:
        """Test error when updating non-existent file."""
        with pytest.raises(FileNotFoundError):
            await file_memory.update_file("/nonexistent/file.txt")


class TestRemoveByPath:
    """Tests for removing files by path."""

    async def test_remove_file(self, file_memory: FileMemory, fixtures_dir: Path) -> None:
        """Test removing a file and its chunks."""
        await file_memory.add_file(fixtures_dir / "sample.txt")
        count_before = await file_memory.count()

        removed = await file_memory.remove_by_path(fixtures_dir / "sample.txt")

        assert removed > 0
        count_after = await file_memory.count()
        assert count_after < count_before

    async def test_remove_nonexistent(self, file_memory: FileMemory) -> None:
        """Test removing non-indexed file."""
        removed = await file_memory.remove_by_path("/nonexistent/file.txt")

        assert removed == 0


class TestAgentMemoryIntegration:
    """Tests for FileMemory integration with AgentMemory."""

    async def test_has_file_property(self, memory_with_file: AgentMemory) -> None:
        """Test has_file property."""
        assert memory_with_file.has_file is True

    async def test_file_property(self, memory_with_file: AgentMemory) -> None:
        """Test file property returns FileMemory."""
        file_mem = memory_with_file.file
        assert isinstance(file_mem, FileMemory)

    async def test_file_in_stats(self, memory_with_file: AgentMemory, fixtures_dir: Path) -> None:
        """Test that file memory is included in stats."""
        await memory_with_file.file.add_file(fixtures_dir / "sample.txt")

        stats = await memory_with_file.get_stats()

        assert "file" in stats
        assert stats["file"] >= 1

    async def test_file_in_clear_all(self, memory_with_file: AgentMemory, fixtures_dir: Path) -> None:
        """Test that clear_all clears file memory."""
        await memory_with_file.file.add_file(fixtures_dir / "sample.txt")

        counts = await memory_with_file.clear_all()

        assert "file" in counts
        count_after = await memory_with_file.file.count()
        assert count_after == 0

    def test_file_without_embeddings(self, backend: InMemoryBackend) -> None:
        """Test that file property raises without embeddings."""
        mem = AgentMemory(
            agent_id="test",
            config=MemoryConfig(
                backend="memory",
                embedding_provider="none",
            ),
            backend=backend,
        )

        assert mem.has_file is False

        with pytest.raises(ValueError, match="requires an embedding provider"):
            _ = mem.file


class TestChunkingIntegration:
    """Tests for chunking configuration with FileMemory."""

    async def test_custom_chunking_config(
        self,
        backend: InMemoryBackend,
        embedding_provider: DummyEmbeddingProvider,
        tmp_path: Path,
    ) -> None:
        """Test using custom chunking configuration."""
        config = ChunkingConfig(chunk_size=50, overlap=10)
        file_memory = FileMemory(
            backend=backend,
            agent_id="test",
            embedding_provider=embedding_provider,
            chunking_config=config,
        )

        # Create a file with content > chunk size
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 200)

        await file_memory.add_file(test_file)

        chunks = await file_memory.get_file_chunks(test_file)
        # Should have multiple chunks due to small chunk size
        assert len(chunks) > 1

    async def test_override_chunking_per_file(
        self,
        file_memory: FileMemory,
        tmp_path: Path,
    ) -> None:
        """Test overriding chunking config per file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 500)

        # Use very small chunks for this file
        config = ChunkingConfig(chunk_size=50, overlap=5)
        await file_memory.add_file(test_file, chunking_config=config)

        chunks = await file_memory.get_file_chunks(test_file)
        assert len(chunks) > 5  # Should have many small chunks
