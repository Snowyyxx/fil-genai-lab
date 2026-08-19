"""Pydantic models for tasks, subtasks and the Zen Cat's AI endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

Source = Literal["bedrock", "mock"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
class Subtask(BaseModel):
    title: str
    estimate_minutes: int = Field(default=15, ge=1, le=480)
    done: bool = False


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    deadline: str | None = None  # ISO date or datetime, kept as a string
    estimate_minutes: int | None = Field(default=None, ge=1, le=1440)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    deadline: str | None = None
    estimate_minutes: int | None = Field(default=None, ge=1, le=1440)
    done: bool | None = None


class SubtaskUpdate(BaseModel):
    done: bool


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    title: str
    notes: str | None = None
    deadline: str | None = None
    estimate_minutes: int | None = None
    done: bool = False
    created_at: str = Field(default_factory=_now)
    subtasks: list[Subtask] = Field(default_factory=list)
    zen_comment: str | None = None


# --------------------------------------------------------------------------- #
# AI endpoints
# --------------------------------------------------------------------------- #
class BreakdownRequest(BaseModel):
    """Break one overwhelming task into bite-sized pieces.

    Pass `task_id` to have the result saved onto that task, or `title` on its
    own to preview a breakdown without touching the list.
    """

    task_id: str | None = None
    title: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)
    max_subtasks: int = Field(default=5, ge=2, le=8)


class BreakdownResponse(BaseModel):
    subtasks: list[Subtask]
    zen_comment: str
    source: Source
    note: str | None = None  # set when we fell back to the offline cat
    task: Task | None = None  # present when the breakdown was saved to a task


class TaskSummary(BaseModel):
    title: str
    done: bool = False
    deadline: str | None = None
    estimate_minutes: int | None = None


class AdviceRequest(BaseModel):
    tasks: list[TaskSummary] = Field(default_factory=list)
    mood: str | None = Field(default=None, max_length=200)


class AdviceResponse(BaseModel):
    wisdom: str
    focus_suggestion: str | None = None
    source: Source
    note: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ai_mode: Literal["bedrock", "mock"]
    model_id: str | None
    region: str
    tasks: int
    detail: str | None = None
