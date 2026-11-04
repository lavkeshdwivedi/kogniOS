"""Skills: instructions kept in markdown files and loaded only when the model asks for them.

A skill is a ``.md`` file with a small frontmatter block (``name`` and ``description``) and a
body. Only the one-line descriptions go into the system prompt, so an agent can know about
dozens of skills for the price of a short index. The body is read when the model calls the
``use_skill`` tool, which keeps the context small until a skill is actually needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def _parse(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    head, sep, body = text[3:].partition("\n---")
    if not sep:
        return None
    meta: dict[str, str] = {}
    for line in head.splitlines():
        key, colon, value = line.partition(":")
        if colon:
            meta[key.strip()] = value.strip()
    if not meta.get("name") or not meta.get("description"):
        return None
    return Skill(meta["name"], meta["description"], body.strip(), path)


class SkillSet:
    def __init__(self, skills: list[Skill]):
        self._skills = {s.name: s for s in skills}

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def index(self) -> str:
        """One line per skill, for the system prompt."""
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        return "Skills you can load with use_skill(name):\n" + "\n".join(lines) if lines else ""

    def get(self, name: str) -> str:
        skill = self._skills.get(name)
        if skill is None:
            return f"Error: unknown skill '{name}'. Known: {', '.join(self._skills) or 'none'}"
        return skill.body

    def tool(self) -> Callable:
        """A ``use_skill`` tool bound to this set, ready to pass in ``Agent(tools=[...])``."""
        skills = self

        def use_skill(name: str) -> str:
            """Load the full instructions of a named skill."""
            return skills.get(name)

        from .tools.registry import tool

        return tool(use_skill)


def load_skills(directory: str | Path) -> SkillSet:
    """Read every ``*.md`` file with valid frontmatter in a directory. Others are skipped."""
    found = [s for p in sorted(Path(directory).glob("*.md")) if (s := _parse(p))]
    return SkillSet(found)
