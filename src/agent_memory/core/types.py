"""Core type definitions for the agent memory framework."""

from enum import StrEnum
from typing import NewType

# A unique identifier for memory records
MemoryId = NewType("MemoryId", str)


class MemoryType(StrEnum):
    """Enumeration of supported memory types."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    CONVERSATION = "conversation"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FILE = "file"
