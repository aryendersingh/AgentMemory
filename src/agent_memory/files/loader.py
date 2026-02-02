"""File loader with automatic extractor selection and chunking."""

from __future__ import annotations

from pathlib import Path

from agent_memory.files.extractors.base import ContentExtractor
from agent_memory.files.extractors.pdf import PDFExtractor, is_pdf_available
from agent_memory.files.extractors.text import TextExtractor
from agent_memory.files.types import (
    ChunkingConfig,
    ChunkingStrategy,
    ExtractedFile,
    FileChunk,
)


class FileLoader:
    """Loads files and extracts their content with automatic chunking.

    The FileLoader selects an appropriate extractor based on file extension
    and provides configurable chunking for large files.

    Example:
        loader = FileLoader()
        extracted = loader.load("./docs/readme.md")
        chunks = loader.chunk(extracted.content)
    """

    def __init__(
        self,
        chunking_config: ChunkingConfig | None = None,
        extractors: list[ContentExtractor] | None = None,
    ) -> None:
        """Initialize the file loader.

        Args:
            chunking_config: Configuration for chunking. Uses defaults if not provided.
            extractors: Custom extractors to use. Uses default set if not provided.
        """
        self._chunking_config = chunking_config or ChunkingConfig()
        self._extractors = extractors or self._default_extractors()

    def _default_extractors(self) -> list[ContentExtractor]:
        """Create the default set of extractors."""
        extractors: list[ContentExtractor] = [TextExtractor()]

        # Add PDF extractor if available
        if is_pdf_available():
            extractors.append(PDFExtractor())

        return extractors

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Get all supported file extensions."""
        extensions: set[str] = set()
        for extractor in self._extractors:
            extensions.update(extractor.supported_extensions)
        return frozenset(extensions)

    def can_load(self, path: str | Path) -> bool:
        """Check if the file can be loaded.

        Args:
            path: Path to the file.

        Returns:
            True if the file can be loaded.
        """
        path = Path(path)
        return any(e.can_extract(path) for e in self._extractors)

    def get_extractor(self, path: str | Path) -> ContentExtractor | None:
        """Get the appropriate extractor for a file.

        Args:
            path: Path to the file.

        Returns:
            The extractor that can handle this file, or None.
        """
        path = Path(path)
        for extractor in self._extractors:
            if extractor.can_extract(path):
                return extractor
        return None

    def load(self, path: str | Path) -> ExtractedFile:
        """Load and extract content from a file.

        Args:
            path: Path to the file.

        Returns:
            ExtractedFile with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If no extractor supports this file type.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extractor = self.get_extractor(path)
        if extractor is None:
            extension = path.suffix.lstrip(".").lower() or "(no extension)"
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported extensions: {sorted(self.supported_extensions)}"
            )

        return extractor.extract(path)

    def chunk(
        self,
        content: str,
        config: ChunkingConfig | None = None,
    ) -> list[FileChunk]:
        """Split content into chunks.

        Args:
            content: The text content to chunk.
            config: Optional override for chunking config.

        Returns:
            List of FileChunk objects.
        """
        cfg = config or self._chunking_config

        if len(content) <= cfg.chunk_size:
            return [
                FileChunk(
                    content=content,
                    index=0,
                    start_position=0,
                    end_position=len(content),
                    total_chunks=1,
                )
            ]

        if cfg.strategy == ChunkingStrategy.CHARACTER:
            return self._chunk_by_character(content, cfg)
        elif cfg.strategy == ChunkingStrategy.LINE:
            return self._chunk_by_line(content, cfg)
        elif cfg.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(content, cfg)
        else:
            # Default to character chunking
            return self._chunk_by_character(content, cfg)

    def _chunk_by_character(
        self,
        content: str,
        config: ChunkingConfig,
    ) -> list[FileChunk]:
        """Chunk content by character count with optional boundary respect."""
        chunks: list[FileChunk] = []
        start = 0
        content_len = len(content)

        while start < content_len:
            end = min(start + config.chunk_size, content_len)

            # Try to find a natural break point if respecting boundaries
            if config.respect_boundaries and end < content_len:
                end = self._find_break_point(content, start, end)

            chunk_content = content[start:end]
            chunks.append(
                FileChunk(
                    content=chunk_content,
                    index=len(chunks),
                    start_position=start,
                    end_position=end,
                )
            )

            # Move start with overlap
            start = end - config.overlap
            if start <= chunks[-1].start_position:
                # Prevent infinite loop
                start = end

        # Update total_chunks for all chunks
        total = len(chunks)
        return [
            FileChunk(
                content=c.content,
                index=c.index,
                start_position=c.start_position,
                end_position=c.end_position,
                total_chunks=total,
            )
            for c in chunks
        ]

    def _chunk_by_line(
        self,
        content: str,
        config: ChunkingConfig,
    ) -> list[FileChunk]:
        """Chunk content by lines."""
        lines = content.split("\n")
        chunks: list[FileChunk] = []
        current_lines: list[str] = []
        current_size = 0
        start_position = 0

        for line in lines:
            line_with_newline = line + "\n"
            line_len = len(line_with_newline)

            if current_size + line_len > config.chunk_size and current_lines:
                # Save current chunk
                chunk_content = "\n".join(current_lines)
                chunks.append(
                    FileChunk(
                        content=chunk_content,
                        index=len(chunks),
                        start_position=start_position,
                        end_position=start_position + len(chunk_content),
                    )
                )

                # Calculate overlap lines
                overlap_size = 0
                overlap_lines: list[str] = []
                for prev_line in reversed(current_lines):
                    if overlap_size + len(prev_line) + 1 <= config.overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_size += len(prev_line) + 1
                    else:
                        break

                start_position += len(chunk_content) + 1 - overlap_size
                current_lines = overlap_lines
                current_size = overlap_size

            current_lines.append(line)
            current_size += line_len

        # Add remaining content
        if current_lines:
            chunk_content = "\n".join(current_lines)
            chunks.append(
                FileChunk(
                    content=chunk_content,
                    index=len(chunks),
                    start_position=start_position,
                    end_position=start_position + len(chunk_content),
                )
            )

        # Update total_chunks
        total = len(chunks)
        return [
            FileChunk(
                content=c.content,
                index=c.index,
                start_position=c.start_position,
                end_position=c.end_position,
                total_chunks=total,
            )
            for c in chunks
        ]

    def _chunk_by_paragraph(
        self,
        content: str,
        config: ChunkingConfig,
    ) -> list[FileChunk]:
        """Chunk content by paragraphs (double newlines)."""
        paragraphs = content.split("\n\n")
        chunks: list[FileChunk] = []
        current_paragraphs: list[str] = []
        current_size = 0
        start_position = 0

        for para in paragraphs:
            para_len = len(para) + 2  # Account for \n\n separator

            if current_size + para_len > config.chunk_size and current_paragraphs:
                chunk_content = "\n\n".join(current_paragraphs)
                chunks.append(
                    FileChunk(
                        content=chunk_content,
                        index=len(chunks),
                        start_position=start_position,
                        end_position=start_position + len(chunk_content),
                    )
                )

                start_position += len(chunk_content) + 2
                current_paragraphs = []
                current_size = 0

            current_paragraphs.append(para)
            current_size += para_len

        # Add remaining content
        if current_paragraphs:
            chunk_content = "\n\n".join(current_paragraphs)
            chunks.append(
                FileChunk(
                    content=chunk_content,
                    index=len(chunks),
                    start_position=start_position,
                    end_position=start_position + len(chunk_content),
                )
            )

        # Update total_chunks
        total = len(chunks)
        return [
            FileChunk(
                content=c.content,
                index=c.index,
                start_position=c.start_position,
                end_position=c.end_position,
                total_chunks=total,
            )
            for c in chunks
        ]

    def _find_break_point(self, content: str, start: int, end: int) -> int:
        """Find a natural break point near the end position.

        Looks for paragraph breaks, sentence ends, or word boundaries.

        Args:
            content: The full content.
            start: Start position of current chunk.
            end: Proposed end position.

        Returns:
            Adjusted end position at a natural break.
        """
        # Look for breaks in reverse order of preference
        search_start = max(start, end - 200)  # Don't look too far back

        # Try paragraph break
        para_break = content.rfind("\n\n", search_start, end)
        if para_break > start:
            return para_break + 2

        # Try sentence end
        for punct in [".\n", "!\n", "?\n", ". ", "! ", "? "]:
            sent_break = content.rfind(punct, search_start, end)
            if sent_break > start:
                return sent_break + len(punct)

        # Try line break
        line_break = content.rfind("\n", search_start, end)
        if line_break > start:
            return line_break + 1

        # Try word boundary (space)
        space = content.rfind(" ", search_start, end)
        if space > start:
            return space + 1

        # No good break found, use original end
        return end

    def load_and_chunk(
        self,
        path: str | Path,
        config: ChunkingConfig | None = None,
    ) -> tuple[ExtractedFile, list[FileChunk]]:
        """Load a file and chunk its content in one operation.

        Args:
            path: Path to the file.
            config: Optional chunking config override.

        Returns:
            Tuple of (ExtractedFile, list of FileChunks).
        """
        extracted = self.load(path)
        chunks = self.chunk(extracted.content, config)
        return extracted, chunks
