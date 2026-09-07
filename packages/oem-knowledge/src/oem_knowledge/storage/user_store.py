"""User store — user-scoped memory I/O wrapper.

Thin wrapper over the user events path for cross-session memory.
No protocol defined yet — this is a convenience wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from oem_knowledge.fs import FileLock

if TYPE_CHECKING:
    from oem_knowledge.engine import KnowledgeEngine


class UserStore:
    """Thin wrapper for user-scoped event persistence."""

    def __init__(self, engine: "KnowledgeEngine"):
        self._engine = engine

    def get_events_path(self) -> Path | None:
        """Return the path to the user events file."""
        from oem_knowledge.services.state import get_user_events_path
        return get_user_events_path()

    def load_events(self) -> list[dict]:
        """Load all user-scoped events."""
        path = self.get_events_path()
        if not path:
            return []
        with FileLock(path.with_suffix(".lock")):
            if not path.exists():
                return []
            events = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return events

    def append_event(self, event: dict) -> None:
        """Append a user-scoped event."""
        path = self.get_events_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(path.with_suffix(".lock")):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
