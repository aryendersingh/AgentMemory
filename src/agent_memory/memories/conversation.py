"""Conversation memory for chat history management."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from agent_memory.core.memory import MemoryRecord
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str
    timestamp: datetime
    metadata: dict[str, Any]


class ConversationMemory(BaseMemory):
    """Memory for managing conversation history.

    Stores messages with role information and provides utilities
    for conversation summarization and context window management.
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
        conversation_id: str | None = None,
        max_messages: int = 100,
        default_ttl: timedelta | None = None,
    ) -> None:
        """Initialize conversation memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: Optional embedding provider.
            conversation_id: Optional ID for this conversation.
            max_messages: Maximum messages to retain.
            default_ttl: Optional TTL for messages.
        """
        super().__init__(backend, agent_id, embedding_provider)
        self._conversation_id = conversation_id or "default"
        self._max_messages = max_messages
        self._default_ttl = default_ttl

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.CONVERSATION

    @property
    def conversation_id(self) -> str:
        """The ID of the current conversation."""
        return self._conversation_id

    async def add_message(
        self,
        role: Role,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryId:
        """Add a message to the conversation.

        Args:
            role: The role of the message sender.
            content: The message content.
            importance: Importance score.
            metadata: Optional additional metadata.

        Returns:
            The ID of the stored message.
        """
        msg_metadata: dict[str, Any] = {
            "role": role,
            "conversation_id": self._conversation_id,
            "message_index": await self._get_next_index(),
        }
        if metadata:
            msg_metadata.update(metadata)

        memory_id = await self.add(
            content=content,
            importance=importance,
            metadata=msg_metadata,
            ttl=self._default_ttl,
        )

        # Trim old messages if needed
        await self._trim_old_messages()

        return memory_id

    async def _get_next_index(self) -> int:
        """Get the next message index for ordering."""
        messages = await self.get_messages(limit=1)
        if not messages:
            return 0
        return messages[0].metadata.get("message_index", 0) + 1

    async def _trim_old_messages(self) -> None:
        """Remove oldest messages if we exceed max_messages."""
        messages = await self.get_messages(limit=self._max_messages + 100)
        if len(messages) > self._max_messages:
            # Delete oldest messages
            to_delete = sorted(
                messages,
                key=lambda m: m.metadata.get("message_index", 0),
            )[: len(messages) - self._max_messages]

            for msg in to_delete:
                await self._backend.delete(msg.id)

    async def get_messages(
        self,
        limit: int = 50,
        include_system: bool = True,
    ) -> Sequence[MemoryRecord]:
        """Get recent messages in chronological order.

        Args:
            limit: Maximum number of messages.
            include_system: Whether to include system messages.

        Returns:
            Sequence of message records, oldest first.
        """
        all_messages = await self.list(limit=limit * 2)

        # Filter by conversation ID
        messages = [
            m for m in all_messages
            if m.metadata.get("conversation_id") == self._conversation_id
        ]

        if not include_system:
            messages = [m for m in messages if m.metadata.get("role") != "system"]

        # Sort by message index (oldest first for conversation order)
        messages = sorted(
            messages,
            key=lambda m: m.metadata.get("message_index", 0),
        )

        return messages[-limit:] if len(messages) > limit else messages

    async def get_last_n_turns(
        self,
        n: int = 5,
    ) -> list[tuple[MemoryRecord | None, MemoryRecord | None]]:
        """Get the last N conversation turns (user + assistant pairs).

        Args:
            n: Number of turns to retrieve.

        Returns:
            List of (user_message, assistant_message) tuples.
        """
        messages = await self.get_messages(limit=n * 2 + 5)

        turns: list[tuple[MemoryRecord | None, MemoryRecord | None]] = []
        current_user: MemoryRecord | None = None

        for msg in messages:
            role = msg.metadata.get("role")
            if role == "user":
                current_user = msg
            elif role == "assistant" and current_user is not None:
                turns.append((current_user, msg))
                current_user = None

        return turns[-n:]

    async def format_history(
        self,
        limit: int = 20,
        format_template: str = "{role}: {content}",
    ) -> str:
        """Format conversation history as a string.

        Args:
            limit: Maximum number of messages.
            format_template: Template for each message.

        Returns:
            Formatted conversation history.
        """
        messages = await self.get_messages(limit=limit)

        formatted = []
        for msg in messages:
            role = msg.metadata.get("role", "unknown")
            formatted.append(format_template.format(role=role, content=msg.content))

        return "\n".join(formatted)

    async def summarize(
        self,
        summarizer: Any | None = None,  # Callable that takes text and returns summary
    ) -> str:
        """Generate a summary of the conversation.

        Args:
            summarizer: Optional callable to generate the summary.
                       If not provided, returns a simple concatenation.

        Returns:
            Summary of the conversation.
        """
        messages = await self.get_messages(limit=100)

        # Build conversation text
        conversation_text = await self.format_history(limit=100)

        if summarizer is not None:
            return await summarizer(conversation_text)

        # Default: return message count summary
        user_count = sum(1 for m in messages if m.metadata.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.metadata.get("role") == "assistant")

        return (
            f"Conversation with {len(messages)} messages "
            f"({user_count} user, {assistant_count} assistant)"
        )

    async def new_conversation(self, conversation_id: str | None = None) -> str:
        """Start a new conversation.

        Args:
            conversation_id: Optional ID for the new conversation.

        Returns:
            The new conversation ID.
        """
        import uuid

        self._conversation_id = conversation_id or str(uuid.uuid4())
        return self._conversation_id

    async def get_conversations(self) -> list[str]:
        """Get all conversation IDs for this agent.

        Returns:
            List of unique conversation IDs.
        """
        all_messages = await self.list(limit=10000)
        conversation_ids = set()

        for msg in all_messages:
            conv_id = msg.metadata.get("conversation_id")
            if conv_id:
                conversation_ids.add(conv_id)

        return list(conversation_ids)

    async def switch_conversation(self, conversation_id: str) -> None:
        """Switch to a different conversation.

        Args:
            conversation_id: The conversation ID to switch to.
        """
        self._conversation_id = conversation_id

    async def delete_conversation(self, conversation_id: str | None = None) -> int:
        """Delete all messages in a conversation.

        Args:
            conversation_id: The conversation to delete (defaults to current).

        Returns:
            Number of messages deleted.
        """
        target_id = conversation_id or self._conversation_id
        all_messages = await self.list(limit=10000)

        count = 0
        for msg in all_messages:
            if msg.metadata.get("conversation_id") == target_id:
                if await self._backend.delete(msg.id):
                    count += 1

        return count
