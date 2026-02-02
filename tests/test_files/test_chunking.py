"""Tests for file chunking functionality."""

import pytest

from agent_memory.files.loader import FileLoader
from agent_memory.files.types import ChunkingConfig, ChunkingStrategy


@pytest.fixture
def loader() -> FileLoader:
    """Create a FileLoader instance."""
    return FileLoader()


class TestCharacterChunking:
    """Tests for character-based chunking."""

    def test_small_content_single_chunk(self, loader: FileLoader) -> None:
        """Test that small content results in a single chunk."""
        content = "This is a short text."
        config = ChunkingConfig(chunk_size=1000, overlap=100)

        chunks = loader.chunk(content, config)

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].index == 0
        assert chunks[0].total_chunks == 1

    def test_large_content_multiple_chunks(self, loader: FileLoader) -> None:
        """Test that large content is split into multiple chunks."""
        content = "A" * 2500
        config = ChunkingConfig(chunk_size=1000, overlap=100, respect_boundaries=False)

        chunks = loader.chunk(content, config)

        assert len(chunks) > 1
        assert all(len(c.content) <= 1000 for c in chunks)

    def test_chunk_overlap(self, loader: FileLoader) -> None:
        """Test that chunks have proper overlap."""
        content = "Word " * 500  # 2500 characters
        config = ChunkingConfig(chunk_size=1000, overlap=200, respect_boundaries=False)

        chunks = loader.chunk(content, config)

        # Check that consecutive chunks have overlapping content
        for i in range(len(chunks) - 1):
            current_end = chunks[i].content[-100:]
            next_start = chunks[i + 1].content[:200]
            # There should be some overlap
            assert current_end in chunks[i + 1].content or next_start[:100] in chunks[i].content

    def test_chunk_indices(self, loader: FileLoader) -> None:
        """Test that chunk indices are correct."""
        content = "X" * 3000
        config = ChunkingConfig(chunk_size=1000, overlap=0, respect_boundaries=False)

        chunks = loader.chunk(content, config)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.total_chunks == len(chunks)

    def test_chunk_positions(self, loader: FileLoader) -> None:
        """Test that chunk positions are tracked."""
        content = "Hello World Test"
        config = ChunkingConfig(chunk_size=1000, overlap=0)

        chunks = loader.chunk(content, config)

        assert chunks[0].start_position == 0
        assert chunks[0].end_position == len(content)


class TestBoundaryRespect:
    """Tests for natural boundary breaking."""

    def test_breaks_at_paragraph(self, loader: FileLoader) -> None:
        """Test that chunking prefers paragraph breaks."""
        content = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph."
        config = ChunkingConfig(chunk_size=50, overlap=0, respect_boundaries=True)

        chunks = loader.chunk(content, config)

        # Should try to break at paragraph boundaries
        assert len(chunks) >= 2

    def test_breaks_at_sentence(self, loader: FileLoader) -> None:
        """Test that chunking prefers sentence breaks."""
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        config = ChunkingConfig(chunk_size=40, overlap=0, respect_boundaries=True)

        chunks = loader.chunk(content, config)

        # Chunks should end at sentence boundaries when possible
        for chunk in chunks[:-1]:  # Except last chunk
            # Should end with punctuation and space or be at a natural break
            assert chunk.content.rstrip().endswith(('.', '!', '?')) or chunk.content.endswith('\n')

    def test_breaks_at_word(self, loader: FileLoader) -> None:
        """Test that chunking avoids breaking words."""
        content = "supercalifragilisticexpialidocious"
        config = ChunkingConfig(chunk_size=50, overlap=0, respect_boundaries=True)

        # Single long word should stay intact if possible
        chunks = loader.chunk(content, config)
        assert chunks[0].content == content


class TestLineChunking:
    """Tests for line-based chunking."""

    def test_line_chunking_basic(self, loader: FileLoader) -> None:
        """Test basic line-based chunking."""
        content = "\n".join([f"Line {i}" for i in range(100)])
        config = ChunkingConfig(
            chunk_size=100,
            overlap=20,
            strategy=ChunkingStrategy.LINE,
        )

        chunks = loader.chunk(content, config)

        assert len(chunks) > 1
        # Each chunk should contain complete lines
        for chunk in chunks:
            lines = chunk.content.split("\n")
            assert all(line.startswith("Line") or line == "" for line in lines)


class TestParagraphChunking:
    """Tests for paragraph-based chunking."""

    def test_paragraph_chunking_basic(self, loader: FileLoader) -> None:
        """Test basic paragraph-based chunking."""
        paragraphs = [f"Paragraph {i} content here." for i in range(10)]
        content = "\n\n".join(paragraphs)
        config = ChunkingConfig(
            chunk_size=100,
            overlap=0,
            strategy=ChunkingStrategy.PARAGRAPH,
        )

        chunks = loader.chunk(content, config)

        assert len(chunks) > 1


class TestChunkingConfig:
    """Tests for ChunkingConfig validation."""

    def test_valid_config(self) -> None:
        """Test creating a valid config."""
        config = ChunkingConfig(chunk_size=1000, overlap=100)
        assert config.chunk_size == 1000
        assert config.overlap == 100

    def test_invalid_chunk_size(self) -> None:
        """Test that negative chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-1, overlap=0)

    def test_zero_chunk_size(self) -> None:
        """Test that zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0, overlap=0)

    def test_invalid_overlap(self) -> None:
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            ChunkingConfig(chunk_size=100, overlap=-1)

    def test_overlap_too_large(self) -> None:
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=150)
