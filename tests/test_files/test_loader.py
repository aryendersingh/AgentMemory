"""Tests for the FileLoader class."""

from pathlib import Path

import pytest

from agent_memory.files.loader import FileLoader
from agent_memory.files.types import ChunkingConfig


@pytest.fixture
def loader() -> FileLoader:
    """Create a FileLoader instance."""
    return FileLoader()


@pytest.fixture
def fixtures_dir() -> Path:
    """Get the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


class TestFileLoader:
    """Tests for FileLoader."""

    def test_load_text_file(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test loading a plain text file."""
        result = loader.load(fixtures_dir / "sample.txt")

        assert result.content is not None
        assert len(result.content) > 0
        assert "sample text file" in result.content
        assert result.metadata.extension == "txt"
        assert result.metadata.name == "sample.txt"

    def test_load_markdown_file(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test loading a markdown file."""
        result = loader.load(fixtures_dir / "sample.md")

        assert result.content is not None
        assert "# Sample Markdown File" in result.content
        assert result.metadata.extension == "md"

    def test_load_python_file(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test loading a Python source file."""
        result = loader.load(fixtures_dir / "sample.py")

        assert result.content is not None
        assert "def greet" in result.content
        assert "class Calculator" in result.content
        assert result.metadata.extension == "py"

    def test_file_not_found(self, loader: FileLoader) -> None:
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_unsupported_file_type(self, loader: FileLoader, tmp_path: Path) -> None:
        """Test error for unsupported file type."""
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(unsupported)

    def test_can_load(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test can_load method."""
        assert loader.can_load(fixtures_dir / "sample.txt")
        assert loader.can_load(fixtures_dir / "sample.md")
        assert loader.can_load(fixtures_dir / "sample.py")
        assert not loader.can_load(Path("/some/file.xyz"))

    def test_supported_extensions(self, loader: FileLoader) -> None:
        """Test that supported extensions are populated."""
        extensions = loader.supported_extensions

        assert "txt" in extensions
        assert "md" in extensions
        assert "py" in extensions
        assert "js" in extensions
        assert "json" in extensions

    def test_metadata_hash(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test that file hash is computed correctly."""
        result = loader.load(fixtures_dir / "sample.txt")

        assert result.metadata.content_hash is not None
        assert len(result.metadata.content_hash) == 64  # SHA-256 hex length

    def test_load_and_chunk(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test load_and_chunk method."""
        extracted, chunks = loader.load_and_chunk(fixtures_dir / "sample.txt")

        assert extracted.content is not None
        assert len(chunks) >= 1
        assert chunks[0].index == 0

    def test_string_path(self, loader: FileLoader, fixtures_dir: Path) -> None:
        """Test that string paths work."""
        path_str = str(fixtures_dir / "sample.txt")
        result = loader.load(path_str)

        assert result.content is not None


class TestEncodingFallback:
    """Tests for encoding fallback behavior."""

    def test_utf8_file(self, loader: FileLoader, tmp_path: Path) -> None:
        """Test loading a UTF-8 encoded file."""
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("Hello, 世界! émojis: 🎉", encoding="utf-8")

        result = loader.load(utf8_file)
        assert "世界" in result.content
        assert "🎉" in result.content

    def test_latin1_file(self, loader: FileLoader, tmp_path: Path) -> None:
        """Test loading a Latin-1 encoded file."""
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_bytes("Caf\xe9".encode("latin-1"))

        result = loader.load(latin1_file)
        assert "Café" in result.content
