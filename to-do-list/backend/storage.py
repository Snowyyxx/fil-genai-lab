"""A tiny thread-safe JSON-file task store.

Deliberately boring: one file, one lock, whole-file rewrite. It keeps local
runs restart-proof without dragging a database into Phase 1. Swap this module
for DynamoDB/Postgres in Phase 2 and nothing else has to change.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from schemas import Task


class TaskStore:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return  # corrupt or unreadable file: start fresh rather than crash
        for item in raw.get("tasks", []):
            try:
                task = Task.model_validate(item)
            except Exception:
                continue
            self._tasks[task.id] = task

    def _flush(self) -> None:
        payload = {"tasks": [t.model_dump() for t in self._tasks.values()]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write so a crash mid-save can't shred the file.
        fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- API ---------------------------------------------------------------
    def list(self) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        # Unfinished first, then by deadline (undated last), then newest first.
        return sorted(
            tasks,
            key=lambda t: (t.done, t.deadline or "9999", t.created_at),
        )

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def add(self, task: Task) -> Task:
        with self._lock:
            self._tasks[task.id] = task
            self._flush()
        return task

    def save(self, task: Task) -> Task:
        return self.add(task)

    def delete(self, task_id: str) -> bool:
        with self._lock:
            existed = self._tasks.pop(task_id, None) is not None
            if existed:
                self._flush()
        return existed

    def count(self) -> int:
        with self._lock:
            return len(self._tasks)
