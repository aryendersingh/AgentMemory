"""Base protocol for file content extractors."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_memory.files.types import ExtractedFile


@runtime_checkable
class ContentExtractor(Protocol):
    """Protocol for file content extractors.

    Extractors are responsible for reading files and extracting
    their text content for indexing.
    """

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Set of file extensions this extractor supports (lowercase, no dot)."""
        ...

    def can_extract(self, path: Path) -> bool:
        """Check if this extractor can handle the given file.

        Args:
            path: Path to the file.

        Returns:
            True if this extractor can handle the file.
        """
        ...

    def extract(self, path: Path) -> ExtractedFile:
        """Extract text content from a file.

        Args:
            path: Path to the file to extract.

        Returns:
            ExtractedFile with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be extracted.
            IOError: If there's an error reading the file.
        """
        ...


class BaseExtractor:
    """Base class for content extractors with common functionality."""

    _supported_extensions: frozenset[str] = frozenset()

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Set of file extensions this extractor supports."""
        return self._supported_extensions

    def can_extract(self, path: Path) -> bool:
        """Check if this extractor can handle the given file.

        Args:
            path: Path to the file.

        Returns:
            True if the file extension is supported.
        """
        extension = path.suffix.lstrip(".").lower()
        return extension in self.supported_extensions
