"""Custom exceptions for the agent memory framework."""


class AgentMemoryError(Exception):
    """Base exception for all agent memory errors."""

    pass


class BackendError(AgentMemoryError):
    """Error raised when a storage backend operation fails."""

    def __init__(self, message: str, backend: str | None = None) -> None:
        self.backend = backend
        super().__init__(f"[{backend}] {message}" if backend else message)


class MemoryNotFoundError(AgentMemoryError):
    """Error raised when a requested memory is not found."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory not found: {memory_id}")


class EmbeddingError(AgentMemoryError):
    """Error raised when embedding generation fails."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}" if provider else message)


class ValidationError(AgentMemoryError):
    """Error raised when validation of memory data fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(f"Validation error for '{field}': {message}" if field else message)


class ConfigurationError(AgentMemoryError):
    """Error raised when configuration is invalid."""

    pass


class ConsolidationError(AgentMemoryError):
    """Error raised when memory consolidation fails."""

    pass
