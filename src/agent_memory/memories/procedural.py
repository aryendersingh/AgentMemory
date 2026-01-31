"""Procedural memory for skills, patterns, and workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_memory.core.memory import MemoryRecord
from agent_memory.core.protocols import EmbeddingProvider, StorageBackend
from agent_memory.core.types import MemoryId, MemoryType
from agent_memory.memories.base import BaseMemory


@dataclass
class Skill:
    """A learned skill or procedure."""

    name: str
    pattern: str
    triggers: list[str]
    success_rate: float
    usage_count: int
    last_used: datetime | None
    metadata: dict[str, Any]


class ProceduralMemory(BaseMemory):
    """Memory for skills, patterns, and learned workflows.

    Procedural memory stores "how to do things" - patterns,
    procedures, and skills that can be retrieved and applied.
    Use this for:
    - Error handling patterns
    - Workflow templates
    - Learned procedures
    - Best practices
    """

    def __init__(
        self,
        backend: StorageBackend,
        agent_id: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        """Initialize procedural memory.

        Args:
            backend: The storage backend to use.
            agent_id: The ID of the agent.
            embedding_provider: Optional embedding provider.
        """
        super().__init__(backend, agent_id, embedding_provider)

    @property
    def memory_type(self) -> MemoryType:
        """The type of this memory."""
        return MemoryType.PROCEDURAL

    async def record_skill(
        self,
        name: str,
        pattern: str,
        triggers: list[str] | None = None,
        importance: float = 0.7,
        category: str | None = None,
        examples: list[str] | None = None,
    ) -> MemoryId:
        """Record a skill or pattern.

        Args:
            name: Name of the skill.
            pattern: The procedure/pattern description.
            triggers: Situations that trigger this skill.
            importance: Importance score.
            category: Optional category (e.g., "error_handling", "optimization").
            examples: Optional example applications.

        Returns:
            The ID of the stored skill.
        """
        metadata: dict[str, Any] = {
            "skill_name": name,
            "skill": True,
            "triggers": triggers or [],
            "usage_count": 0,
            "success_count": 0,
        }

        if category:
            metadata["category"] = category
        if examples:
            metadata["examples"] = examples

        return await self.add(
            content=pattern,
            importance=importance,
            metadata=metadata,
            ttl=None,  # Skills are permanent
        )

    async def record_workflow(
        self,
        name: str,
        steps: list[str],
        triggers: list[str] | None = None,
        importance: float = 0.7,
    ) -> MemoryId:
        """Record a multi-step workflow.

        Args:
            name: Name of the workflow.
            steps: Ordered list of steps.
            triggers: Situations that trigger this workflow.
            importance: Importance score.

        Returns:
            The ID of the stored workflow.
        """
        # Format steps as numbered list
        pattern = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))

        metadata: dict[str, Any] = {
            "skill_name": name,
            "workflow": True,
            "skill": True,
            "step_count": len(steps),
            "steps": steps,
            "triggers": triggers or [],
            "usage_count": 0,
            "success_count": 0,
        }

        return await self.add(
            content=pattern,
            importance=importance,
            metadata=metadata,
            ttl=None,
        )

    async def get_skill(self, name: str) -> MemoryRecord | None:
        """Get a skill by name.

        Args:
            name: Name of the skill.

        Returns:
            The skill record if found, None otherwise.
        """
        skills = await self.list(limit=1000)
        for skill in skills:
            if skill.metadata.get("skill_name") == name:
                return skill
        return None

    async def get_skills_by_category(
        self,
        category: str,
        limit: int = 50,
    ) -> Sequence[MemoryRecord]:
        """Get skills in a category.

        Args:
            category: The category to filter by.
            limit: Maximum number of results.

        Returns:
            Sequence of skills in the category.
        """
        all_skills = await self.list(limit=limit * 2)
        return [
            s for s in all_skills
            if s.metadata.get("category") == category
        ][:limit]

    async def find_applicable_skills(
        self,
        situation: str,
        limit: int = 5,
    ) -> Sequence[MemoryRecord]:
        """Find skills applicable to a situation.

        Uses semantic search if embedding provider is available,
        otherwise falls back to trigger matching.

        Args:
            situation: Description of the current situation.
            limit: Maximum number of skills to return.

        Returns:
            Sequence of applicable skills.
        """
        # Try semantic search first
        if self._embedding_provider is not None:
            try:
                scored = await self.search_similar(
                    query=situation,
                    limit=limit,
                    min_score=0.3,
                )
                return [s.memory for s in scored]
            except Exception:
                pass  # Fall back to trigger matching

        # Fall back to trigger keyword matching
        all_skills = await self.list(limit=500)
        situation_lower = situation.lower()

        matches: list[tuple[MemoryRecord, int]] = []
        for skill in all_skills:
            triggers = skill.metadata.get("triggers", [])
            match_count = sum(
                1 for trigger in triggers
                if trigger.lower() in situation_lower
            )
            if match_count > 0:
                matches.append((skill, match_count))

        # Sort by match count
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches[:limit]]

    async def record_skill_usage(
        self,
        skill_id: MemoryId,
        success: bool = True,
    ) -> bool:
        """Record that a skill was used.

        Args:
            skill_id: The ID of the skill that was used.
            success: Whether the usage was successful.

        Returns:
            True if updated, False if skill not found.
        """
        skill = await self._backend.retrieve(skill_id)
        if skill is None:
            return False

        usage_count = skill.metadata.get("usage_count", 0) + 1
        success_count = skill.metadata.get("success_count", 0) + (1 if success else 0)

        new_metadata = {
            **skill.metadata,
            "usage_count": usage_count,
            "success_count": success_count,
            "last_used": datetime.utcnow().isoformat(),
        }

        updated = MemoryRecord(
            id=skill.id,
            content=skill.content,
            memory_type=skill.memory_type,
            importance=skill.importance,
            created_at=skill.created_at,
            accessed_at=datetime.utcnow(),
            access_count=skill.access_count + 1,
            metadata=new_metadata,
            embedding=skill.embedding,
            agent_id=skill.agent_id,
            ttl=skill.ttl,
        )

        return await self._backend.update(updated)

    async def get_most_used_skills(
        self,
        limit: int = 10,
    ) -> Sequence[MemoryRecord]:
        """Get the most frequently used skills.

        Args:
            limit: Maximum number of skills.

        Returns:
            Skills sorted by usage count.
        """
        all_skills = await self.list(limit=500)
        sorted_skills = sorted(
            all_skills,
            key=lambda s: s.metadata.get("usage_count", 0),
            reverse=True,
        )
        return sorted_skills[:limit]

    async def get_most_successful_skills(
        self,
        min_uses: int = 3,
        limit: int = 10,
    ) -> Sequence[MemoryRecord]:
        """Get skills with the highest success rates.

        Args:
            min_uses: Minimum usage count to be included.
            limit: Maximum number of skills.

        Returns:
            Skills sorted by success rate.
        """
        all_skills = await self.list(limit=500)

        # Filter by minimum uses
        qualified = [
            s for s in all_skills
            if s.metadata.get("usage_count", 0) >= min_uses
        ]

        # Calculate success rate and sort
        def success_rate(skill: MemoryRecord) -> float:
            usage = skill.metadata.get("usage_count", 0)
            success = skill.metadata.get("success_count", 0)
            return success / usage if usage > 0 else 0

        sorted_skills = sorted(qualified, key=success_rate, reverse=True)
        return sorted_skills[:limit]

    async def add_trigger(self, skill_id: MemoryId, trigger: str) -> bool:
        """Add a trigger to an existing skill.

        Args:
            skill_id: The ID of the skill.
            trigger: The trigger to add.

        Returns:
            True if updated, False if skill not found.
        """
        skill = await self._backend.retrieve(skill_id)
        if skill is None:
            return False

        triggers = skill.metadata.get("triggers", [])
        if trigger not in triggers:
            triggers = triggers + [trigger]

        updated = skill.with_metadata(triggers=triggers)
        return await self._backend.update(updated)

    async def get_skill_as_object(self, skill_id: MemoryId) -> Skill | None:
        """Get a skill as a Skill object.

        Args:
            skill_id: The ID of the skill.

        Returns:
            Skill object if found, None otherwise.
        """
        record = await self._backend.retrieve(skill_id)
        if record is None:
            return None

        usage_count = record.metadata.get("usage_count", 0)
        success_count = record.metadata.get("success_count", 0)

        last_used_str = record.metadata.get("last_used")
        last_used = (
            datetime.fromisoformat(last_used_str)
            if last_used_str
            else None
        )

        return Skill(
            name=record.metadata.get("skill_name", "unnamed"),
            pattern=record.content,
            triggers=record.metadata.get("triggers", []),
            success_rate=success_count / usage_count if usage_count > 0 else 0.0,
            usage_count=usage_count,
            last_used=last_used,
            metadata={
                k: v
                for k, v in record.metadata.items()
                if k not in {
                    "skill_name",
                    "triggers",
                    "usage_count",
                    "success_count",
                    "last_used",
                    "skill",
                }
            },
        )
