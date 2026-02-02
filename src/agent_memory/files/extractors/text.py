"""Text and code file extractor."""

from __future__ import annotations

from pathlib import Path

from agent_memory.files.extractors.base import BaseExtractor
from agent_memory.files.types import ExtractedFile, FileMetadata


class TextExtractor(BaseExtractor):
    """Extractor for plain text and code files.

    Supports various text formats including plain text, markdown,
    and common programming languages with encoding fallback.
    """

    _supported_extensions: frozenset[str] = frozenset([
        # Plain text
        "txt",
        "text",
        "log",
        # Markdown and documentation
        "md",
        "markdown",
        "rst",
        "adoc",
        # Web
        "html",
        "htm",
        "css",
        "scss",
        "sass",
        "less",
        # JavaScript/TypeScript
        "js",
        "jsx",
        "ts",
        "tsx",
        "mjs",
        "cjs",
        # Python
        "py",
        "pyi",
        "pyw",
        # Other languages
        "java",
        "kt",
        "kts",
        "scala",
        "groovy",
        "c",
        "h",
        "cpp",
        "hpp",
        "cc",
        "cxx",
        "cs",
        "go",
        "rs",
        "rb",
        "php",
        "swift",
        "m",
        "mm",
        "r",
        "R",
        "pl",
        "pm",
        "lua",
        "sh",
        "bash",
        "zsh",
        "fish",
        "ps1",
        "bat",
        "cmd",
        # Data formats
        "json",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "conf",
        "xml",
        "csv",
        "tsv",
        # Config files
        "env",
        "gitignore",
        "dockerignore",
        "editorconfig",
        # SQL
        "sql",
        # Other
        "makefile",
        "dockerfile",
        "vagrantfile",
    ])

    # Encodings to try in order
    _encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]

    def extract(self, path: Path) -> ExtractedFile:
        """Extract text content from a file.

        Args:
            path: Path to the file to extract.

        Returns:
            ExtractedFile with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be decoded.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Read file as bytes first
        content_bytes = path.read_bytes()

        # Try to decode with different encodings
        content = self._decode_content(content_bytes, path)

        # Create metadata
        metadata = FileMetadata.from_path(path, content_bytes)

        return ExtractedFile(
            content=content,
            metadata=metadata,
        )

    def _decode_content(self, content_bytes: bytes, path: Path) -> str:
        """Try to decode content with multiple encodings.

        Args:
            content_bytes: Raw file content.
            path: Path for error messages.

        Returns:
            Decoded string content.

        Raises:
            ValueError: If no encoding works.
        """
        last_error: Exception | None = None

        for encoding in self._encodings:
            try:
                return content_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue

        raise ValueError(
            f"Could not decode file {path} with any supported encoding. "
            f"Last error: {last_error}"
        )
