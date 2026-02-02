"""Type definitions for file handling in the memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class ChunkingStrategy(StrEnum):
    """Strategy for chunking file content."""

    CHARACTER = "character"
    LINE = "line"
    PARAGRAPH = "paragraph"


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """Metadata about a file.

    Attributes:
        path: Absolute path to the file.
        name: File name without directory.
        extension: File extension (lowercase, without dot).
        size: File size in bytes.
        content_hash: SHA-256 hash of the file content.
        modified_at: Last modification time of the file.
        mime_type: MIME type of the file (if known).
    """

    path: str
    name: str
    extension: str
    size: int
    content_hash: str
    modified_at: datetime
    mime_type: str | None = None

    @classmethod
    def from_path(cls, path: Path, content: bytes) -> "FileMetadata":
        """Create FileMetadata from a path and content.

        Args:
            path: Path to the file.
            content: File content as bytes.

        Returns:
            FileMetadata instance.
        """
        import hashlib

        stat = path.stat()
        content_hash = hashlib.sha256(content).hexdigest()

        return cls(
            path=str(path.resolve()),
            name=path.name,
            extension=path.suffix.lstrip(".").lower(),
            size=stat.st_size,
            content_hash=content_hash,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage in metadata."""
        return {
            "file_path": self.path,
            "file_name": self.name,
            "file_extension": self.extension,
            "file_size": self.size,
            "file_hash": self.content_hash,
            "file_modified_at": self.modified_at.isoformat(),
            "file_mime_type": self.mime_type,
        }


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration for chunking file content.

    Attributes:
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.
        strategy: Chunking strategy to use.
        respect_boundaries: Whether to try to break at natural boundaries.
    """

    chunk_size: int = 1000
    overlap: int = 200
    strategy: ChunkingStrategy = ChunkingStrategy.CHARACTER
    respect_boundaries: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")


@dataclass(frozen=True, slots=True)
class FileChunk:
    """A chunk of file content.

    Attributes:
        content: The text content of the chunk.
        index: Zero-based index of this chunk.
        start_position: Starting character position in the original content.
        end_position: Ending character position in the original content.
        total_chunks: Total number of chunks for this file.
    """

    content: str
    index: int
    start_position: int
    end_position: int
    total_chunks: int = 1

    def to_metadata(self) -> dict[str, Any]:
        """Convert chunk info to metadata for storage."""
        return {
            "chunk_index": self.index,
            "chunk_start": self.start_position,
            "chunk_end": self.end_position,
            "total_chunks": self.total_chunks,
        }


@dataclass(slots=True)
class ExtractedFile:
    """Result of extracting content from a file.

    Attributes:
        content: The extracted text content.
        metadata: Metadata about the file.
        extra: Additional extraction-specific data.
    """

    content: str
    metadata: FileMetadata
    extra: dict[str, Any] = field(default_factory=dict)
