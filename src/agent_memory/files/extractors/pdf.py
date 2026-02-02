"""PDF file extractor using PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_memory.files.extractors.base import BaseExtractor
from agent_memory.files.types import ExtractedFile, FileMetadata

if TYPE_CHECKING:
    import fitz


# Check if pymupdf is available
_PYMUPDF_AVAILABLE = False
try:
    import fitz  # noqa: F811

    _PYMUPDF_AVAILABLE = True
except ImportError:
    pass


def is_pdf_available() -> bool:
    """Check if PDF extraction is available."""
    return _PYMUPDF_AVAILABLE


class PDFExtractor(BaseExtractor):
    """Extractor for PDF files using PyMuPDF.

    Requires the optional pymupdf dependency:
        pip install agent-memory[pdf]
    """

    _supported_extensions: frozenset[str] = frozenset(["pdf"])

    def __init__(self) -> None:
        """Initialize the PDF extractor.

        Raises:
            ImportError: If pymupdf is not installed.
        """
        if not _PYMUPDF_AVAILABLE:
            raise ImportError(
                "PDF extraction requires pymupdf package. "
                "Install with: pip install agent-memory[pdf]"
            )

    def extract(self, path: Path) -> ExtractedFile:
        """Extract text content from a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            ExtractedFile with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be processed.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Read file as bytes for hash calculation
        content_bytes = path.read_bytes()

        # Extract text from PDF
        content, extra = self._extract_pdf_text(path)

        # Create metadata
        metadata = FileMetadata.from_path(path, content_bytes)

        return ExtractedFile(
            content=content,
            metadata=metadata,
            extra=extra,
        )

    def _extract_pdf_text(self, path: Path) -> tuple[str, dict[str, Any]]:
        """Extract text from PDF using PyMuPDF.

        Args:
            path: Path to the PDF file.

        Returns:
            Tuple of (extracted text, extra metadata).
        """
        import fitz

        doc = fitz.open(str(path))
        try:
            pages: list[str] = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text)

            content = "\n\n".join(pages)

            extra = {
                "page_count": len(doc),
                "pdf_title": doc.metadata.get("title", ""),
                "pdf_author": doc.metadata.get("author", ""),
                "pdf_subject": doc.metadata.get("subject", ""),
                "pdf_keywords": doc.metadata.get("keywords", ""),
            }

            return content, extra
        finally:
            doc.close()
