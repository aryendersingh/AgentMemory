"""File handling for the agent memory system."""

from agent_memory.files.extractors import (
    BaseExtractor,
    ContentExtractor,
    PDFExtractor,
    TextExtractor,
    is_pdf_available,
)
from agent_memory.files.loader import FileLoader
from agent_memory.files.types import (
    ChunkingConfig,
    ChunkingStrategy,
    ExtractedFile,
    FileChunk,
    FileMetadata,
)

__all__ = [
    # Types
    "ChunkingConfig",
    "ChunkingStrategy",
    "ExtractedFile",
    "FileChunk",
    "FileMetadata",
    # Loader
    "FileLoader",
    # Extractors
    "BaseExtractor",
    "ContentExtractor",
    "PDFExtractor",
    "TextExtractor",
    "is_pdf_available",
]
