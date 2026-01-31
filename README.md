# Agent Memory Framework

A developer-friendly Python API framework for AI agent memory with pluggable storage backends, designed for autonomous agents.

## Features

- **Multiple Memory Types**: Short-term, long-term, conversation, episodic, semantic, and procedural memory
- **Pluggable Backends**: In-memory (included), Redis, PostgreSQL, ChromaDB, Pinecone
- **Vector Search**: Semantic memory with embedding-based similarity search
- **Async-First**: All operations are async with sync wrappers available
- **Type-Safe**: Full type hints and protocol-based interfaces

## Installation

```bash
pip install agent-memory

# With optional backends
pip install agent-memory[redis]
pip install agent-memory[postgres]
pip install agent-memory[chroma]
pip install agent-memory[openai]
pip install agent-memory[all]
```

## Quick Start

```python
import asyncio
from agent_memory import AgentMemory, MemoryConfig

async def main():
    # Initialize with in-memory backend and dummy embeddings
    config = MemoryConfig(
        backend="memory",
        embedding_provider="dummy",  # Use "openai" for production
    )

    async with AgentMemory(agent_id="my-agent", config=config) as memory:
        # Short-term memory (session context)
        await memory.short_term.add("User prefers dark mode", importance=0.7)

        # Long-term memory (persistent facts)
        await memory.long_term.store_fact("User's name is Alice", tags=["personal"])

        # Conversation memory
        await memory.conversation.add_message(role="user", content="Hello!")
        await memory.conversation.add_message(role="assistant", content="Hi there!")

        # Episodic memory (events)
        await memory.episodic.record_event(
            event_type="task_completed",
            description="Finished code review",
            outcome="success",
        )

        # Semantic memory (vector search)
        await memory.semantic.add("Python was created by Guido van Rossum")
        results = await memory.semantic.search("Who made Python?", limit=5)

        # Procedural memory (skills)
        await memory.procedural.record_skill(
            name="error_handling",
            pattern="When API returns 429, retry with exponential backoff",
            triggers=["rate limit", "429"],
        )

asyncio.run(main())
```

## Memory Types

| Memory Type | Purpose | TTL | Persistence |
|-------------|---------|-----|-------------|
| **Short-term** | Current task/session context | Minutes-hours | Session only |
| **Long-term** | Persistent facts, preferences | None | Permanent |
| **Conversation** | Chat history with summarization | Configurable | Per-conversation |
| **Episodic** | Events with temporal context | Optional | Permanent |
| **Semantic** | Vector-embedded knowledge | None | Permanent |
| **Procedural** | Learned skills/workflows | None | Permanent |

## API Reference

### AgentMemory

The main facade class providing unified access to all memory types.

```python
memory = AgentMemory(
    agent_id="my-agent",
    config=MemoryConfig(...),
)

# Access memory types
memory.short_term   # ShortTermMemory
memory.long_term    # LongTermMemory
memory.conversation # ConversationMemory
memory.episodic     # EpisodicMemory
memory.semantic     # SemanticMemory (requires embedding provider)
memory.procedural   # ProceduralMemory
```

### ShortTermMemory

Working memory with automatic expiration.

```python
await memory.short_term.add("context", importance=0.7)
await memory.short_term.add_context("task state", context_type="task")
await memory.short_term.set_working_memory("key", "value")
value = await memory.short_term.get_working_memory("key")
await memory.short_term.cleanup_expired()
```

### LongTermMemory

Persistent storage for facts and preferences.

```python
await memory.long_term.store_fact("fact", tags=["tag1", "tag2"])
await memory.long_term.store_preference("category", "preference")
facts = await memory.long_term.get_facts(tags=["tag1"])
prefs = await memory.long_term.get_preferences("category")
```

### ConversationMemory

Chat history with turn management.

```python
await memory.conversation.add_message(role="user", content="Hello")
messages = await memory.conversation.get_messages(limit=10)
turns = await memory.conversation.get_last_n_turns(5)
history = await memory.conversation.format_history()
await memory.conversation.new_conversation()
```

### EpisodicMemory

Event recording with temporal queries.

```python
await memory.episodic.record_event(
    event_type="action",
    description="Did something",
    outcome="success",
)
events = await memory.episodic.get_events_by_type("action")
recent = await memory.episodic.get_recent_events(hours=24)
```

### SemanticMemory

Vector-based semantic search.

```python
await memory.semantic.add("knowledge content")
results = await memory.semantic.search("query", limit=10)
related = await memory.semantic.find_related(memory_id, limit=5)
```

### ProceduralMemory

Skills and workflow storage.

```python
await memory.procedural.record_skill(
    name="skill_name",
    pattern="How to do something",
    triggers=["trigger1", "trigger2"],
)
skills = await memory.procedural.find_applicable_skills("situation")
await memory.procedural.record_skill_usage(skill_id, success=True)
```

## Configuration

```python
from agent_memory import MemoryConfig

config = MemoryConfig(
    backend="memory",           # "memory", "redis", "postgres", "chroma"
    embedding_provider="dummy", # "none", "dummy", "openai"
    backend_config={},          # Backend-specific settings
    embedding_config={},        # Embedding provider settings
    short_term_ttl=timedelta(hours=1),
    conversation_max_messages=100,
    enable_scoring=True,
    enable_decay=True,
)
```

## Running Examples

```bash
PYTHONPATH=src python3 examples/basic_usage.py
```

## License

MIT
