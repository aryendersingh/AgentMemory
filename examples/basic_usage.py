"""Basic usage example for the Agent Memory Framework.

This example demonstrates the core features of the memory system.
Run with: python examples/basic_usage.py
"""

import asyncio

from agent_memory import AgentMemory, MemoryConfig


async def main():
    """Demonstrate basic usage of the agent memory framework."""

    # Initialize the memory system with in-memory backend and dummy embeddings
    config = MemoryConfig(
        backend="memory",
        embedding_provider="dummy",  # Use "openai" for production
    )

    async with AgentMemory(agent_id="demo-agent", config=config) as memory:
        print("=== Agent Memory Framework Demo ===\n")

        # ---- Short-term Memory ----
        print("1. Short-term Memory (session context)")
        await memory.short_term.add("User prefers dark mode", importance=0.7)
        await memory.short_term.add_context(
            "Current task: writing documentation",
            context_type="task_state",
        )
        await memory.short_term.set_working_memory("current_file", "README.md")

        working = await memory.short_term.get_working_memory("current_file")
        print(f"   Working memory 'current_file': {working.content if working else None}")
        print()

        # ---- Long-term Memory ----
        print("2. Long-term Memory (persistent facts)")
        await memory.long_term.store_fact(
            "User's name is Alice",
            tags=["personal", "identity"],
        )
        await memory.long_term.store_preference(
            category="communication",
            preference="Prefers concise responses",
        )

        facts = await memory.long_term.get_facts()
        print(f"   Stored {len(facts)} fact(s)")
        for fact in facts:
            print(f"   - {fact.content}")
        print()

        # ---- Conversation Memory ----
        print("3. Conversation Memory (chat history)")
        await memory.conversation.add_message(role="user", content="Hello, how are you?")
        await memory.conversation.add_message(
            role="assistant",
            content="I'm doing well! How can I help you today?",
        )
        await memory.conversation.add_message(
            role="user",
            content="Can you help me write some code?",
        )

        history = await memory.conversation.format_history()
        print("   Recent conversation:")
        for line in history.split("\n"):
            print(f"   {line}")
        print()

        # ---- Episodic Memory ----
        print("4. Episodic Memory (events)")
        await memory.episodic.record_event(
            event_type="task_completed",
            description="Successfully generated Python code for data processing",
            outcome="success",
            importance=0.8,
        )
        await memory.episodic.record_action(
            action="Executed unit tests",
            result="All 15 tests passed",
            success=True,
        )

        events = await memory.episodic.get_recent_events(hours=24)
        print(f"   Recorded {len(events)} event(s)")
        for event in events:
            print(f"   - [{event.metadata.get('event_type')}] {event.content[:50]}...")
        print()

        # ---- Procedural Memory ----
        print("5. Procedural Memory (skills)")
        skill_id = await memory.procedural.record_skill(
            name="error_handling",
            pattern="When an API returns 429, wait and retry with exponential backoff",
            triggers=["rate limit", "429", "too many requests"],
            category="api",
        )

        await memory.procedural.record_workflow(
            name="code_review",
            steps=[
                "Check for syntax errors",
                "Verify test coverage",
                "Review for security issues",
                "Check code style",
            ],
            triggers=["review", "pull request"],
        )

        # Simulate using the skill
        await memory.procedural.record_skill_usage(skill_id, success=True)

        skills = await memory.procedural.find_applicable_skills("API rate limit error")
        print(f"   Found {len(skills)} applicable skill(s) for 'API rate limit error'")
        for skill in skills:
            print(f"   - {skill.metadata.get('skill_name')}: {skill.content[:50]}...")
        print()

        # ---- Semantic Memory ----
        print("6. Semantic Memory (vector search)")
        await memory.semantic.add("Python was created by Guido van Rossum in 1991")
        await memory.semantic.add("JavaScript was created by Brendan Eich in 1995")
        await memory.semantic.add("Rust was developed by Mozilla starting in 2010")

        results = await memory.semantic.search("Who created Python?", limit=2)
        print("   Search for 'Who created Python?':")
        for scored in results:
            print(f"   - Score {scored.score:.3f}: {scored.memory.content}")
        print()

        # ---- Statistics ----
        print("7. Memory Statistics")
        stats = await memory.get_stats()
        print(f"   Agent ID: {stats['agent_id']}")
        print(f"   Short-term: {stats['short_term']} memories")
        print(f"   Long-term: {stats['long_term']} memories")
        print(f"   Conversation: {stats['conversation']} messages")
        print(f"   Episodic: {stats['episodic']} events")
        print(f"   Procedural: {stats['procedural']} skills")
        print(f"   Semantic: {stats.get('semantic', 0)} documents")
        print()

        print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
