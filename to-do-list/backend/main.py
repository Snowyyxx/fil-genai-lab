"""Meowstermind API - a cozy Zen-cat task master.

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import zen_cat
from config import settings
from schemas import (
    AdviceRequest,
    AdviceResponse,
    BreakdownRequest,
    BreakdownResponse,
    HealthResponse,
    Subtask,
    SubtaskUpdate,
    Task,
    TaskCreate,
    TaskSummary,
    TaskUpdate,
)
from storage import TaskStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Meowstermind",
    description="A peaceful Zen cat who keeps your tasks bite-sized.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = TaskStore(settings.data_file)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report whether the cat is thinking with Bedrock or dreaming offline."""
    if not settings.force_mock:
        # Resolve once (cached) so this answer reflects real Bedrock reachability
        # instead of optimistically claiming "live" until the first AI call.
        zen_cat.brain.resolve_model_id()
    return HealthResponse(
        status="ok",
        ai_mode="mock" if settings.force_mock else "bedrock",
        model_id=zen_cat.brain.model_id,
        region=settings.aws_region,
        tasks=store.count(),
        detail=zen_cat.brain.last_error,
    )


@app.get("/ai/models", tags=["ai"])
def list_models() -> dict:
    """Which open-weight models this AWS account can invoke, best first.

    Useful the first time you connect: pick one and pin it with BEDROCK_MODEL_ID.
    """
    if settings.force_mock:
        return {"mode": "mock", "inference_profiles": [], "on_demand_models": []}
    try:
        return {"mode": "bedrock", "selected": zen_cat.brain.model_id, **zen_cat.brain.available_models()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Bedrock: {exc}") from exc


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@app.get("/tasks", response_model=list[Task], tags=["tasks"])
def list_tasks() -> list[Task]:
    return store.list()


@app.post("/tasks", response_model=Task, status_code=201, tags=["tasks"])
def create_task(payload: TaskCreate) -> Task:
    return store.add(Task(**payload.model_dump()))


@app.get("/tasks/{task_id}", response_model=Task, tags=["tasks"])
def get_task(task_id: str) -> Task:
    return _require(task_id)


@app.patch("/tasks/{task_id}", response_model=Task, tags=["tasks"])
def update_task(task_id: str, payload: TaskUpdate) -> Task:
    task = _require(task_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    return store.save(task)


@app.delete("/tasks/{task_id}", status_code=204, tags=["tasks"])
def delete_task(task_id: str) -> None:
    if not store.delete(task_id):
        raise HTTPException(status_code=404, detail="No such task")


@app.patch("/tasks/{task_id}/subtasks/{index}", response_model=Task, tags=["tasks"])
def update_subtask(task_id: str, index: int, payload: SubtaskUpdate) -> Task:
    task = _require(task_id)
    if not 0 <= index < len(task.subtasks):
        raise HTTPException(status_code=404, detail="No such subtask")
    task.subtasks[index].done = payload.done
    return store.save(task)


# --------------------------------------------------------------------------- #
# The Zen Cat
# --------------------------------------------------------------------------- #
@app.post("/ai/breakdown", response_model=BreakdownResponse, tags=["ai"])
def ai_breakdown(payload: BreakdownRequest) -> BreakdownResponse:
    """Ask Meowstermind to slice a big task into bite-sized subtasks.

    Send `task_id` to save the result onto that task, or just `title` to preview.
    """
    task: Task | None = None
    if payload.task_id:
        task = _require(payload.task_id)
        title = payload.title or task.title
        notes = payload.notes or task.notes
        payload = payload.model_copy(update={"notes": notes})
    elif payload.title:
        title = payload.title
    else:
        raise HTTPException(status_code=422, detail="Provide either task_id or title")

    result = zen_cat.breakdown(payload, title)

    if task is not None:
        task.subtasks = [Subtask(**s.model_dump()) for s in result.subtasks]
        task.zen_comment = result.zen_comment
        result.task = store.save(task)
    return result


@app.post("/ai/advice", response_model=AdviceResponse, tags=["ai"])
def ai_advice(payload: AdviceRequest | None = None) -> AdviceResponse:
    """Daily wisdom / prioritisation from the Zen Cat.

    With an empty body the cat reads the saved task list itself.
    """
    req = payload or AdviceRequest()
    if not req.tasks:
        # model_copy(update=...) skips validation, so build the models directly.
        req.tasks = [
            TaskSummary(
                title=t.title,
                done=t.done,
                deadline=t.deadline,
                estimate_minutes=t.estimate_minutes,
            )
            for t in store.list()
        ]
    return zen_cat.advice(req)


# --------------------------------------------------------------------------- #
# Helpers + static frontend
# --------------------------------------------------------------------------- #
def _require(task_id: str) -> Task:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="No such task")
    return task


if settings.frontend_dir.is_dir():
    # Serving the UI from the API keeps local dev to one command and makes the
    # Phase 2 container a single image.
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.frontend_dir / "index.html")

    app.mount("/app", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
