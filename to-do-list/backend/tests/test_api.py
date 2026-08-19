"""End-to-end API tests. They run in offline (mock) mode - no AWS needed.

    cd backend && MEOW_FORCE_MOCK=1 pytest -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# Force the offline cat and an isolated data file *before* importing the app.
os.environ["MEOW_FORCE_MOCK"] = "1"
os.environ.setdefault("MEOW_DATA_FILE", str(BACKEND / "tests" / ".tasks-test.json"))


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    import main

    Path(os.environ["MEOW_DATA_FILE"]).unlink(missing_ok=True)
    main.store = main.TaskStore(Path(os.environ["MEOW_DATA_FILE"]))
    with TestClient(main.app) as c:
        yield c
    Path(os.environ["MEOW_DATA_FILE"]).unlink(missing_ok=True)


def test_health_reports_mock_mode(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ai_mode"] == "mock"


def test_create_list_and_complete_a_task(client):
    created = client.post(
        "/tasks",
        json={"title": "Write the thesis intro", "deadline": "2026-09-01", "estimate_minutes": 90},
    )
    assert created.status_code == 201
    task = created.json()
    assert task["title"] == "Write the thesis intro"
    assert task["done"] is False

    assert [t["id"] for t in client.get("/tasks").json()] == [task["id"]]

    done = client.patch(f"/tasks/{task['id']}", json={"done": True}).json()
    assert done["done"] is True

    assert client.delete(f"/tasks/{task['id']}").status_code == 204
    assert client.get("/tasks").json() == []


def test_blank_title_is_rejected(client):
    assert client.post("/tasks", json={"title": "   "}).status_code == 422


def test_breakdown_preview_does_not_touch_the_list(client):
    res = client.post("/ai/breakdown", json={"title": "Plan the trip", "max_subtasks": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["source"] == "mock"
    assert 1 <= len(body["subtasks"]) <= 3
    assert body["zen_comment"]
    assert body["task"] is None
    assert client.get("/tasks").json() == []


def test_breakdown_saves_subtasks_onto_a_task(client):
    task = client.post("/tasks", json={"title": "Clean the whole flat"}).json()

    body = client.post("/ai/breakdown", json={"task_id": task["id"], "max_subtasks": 4}).json()
    assert body["task"]["id"] == task["id"]
    assert len(body["task"]["subtasks"]) == 4
    assert body["task"]["zen_comment"]

    toggled = client.patch(f"/tasks/{task['id']}/subtasks/0", json={"done": True}).json()
    assert toggled["subtasks"][0]["done"] is True

    assert client.patch(f"/tasks/{task['id']}/subtasks/99", json={"done": True}).status_code == 404


def test_breakdown_needs_a_title_or_task_id(client):
    assert client.post("/ai/breakdown", json={}).status_code == 422


def test_advice_reads_the_saved_list_when_body_is_empty(client):
    client.post("/tasks", json={"title": "Renew the passport", "deadline": "2026-08-20"})
    client.post("/tasks", json={"title": "Water the plants", "estimate_minutes": 5})

    body = client.post("/ai/advice", json={}).json()
    assert body["source"] == "mock"
    assert body["wisdom"]
    assert "Renew the passport" in body["focus_suggestion"]  # nearest deadline wins


def test_advice_on_an_empty_list(client):
    body = client.post("/ai/advice", json={}).json()
    assert body["focus_suggestion"] is None


def test_unknown_task_is_404(client):
    assert client.get("/tasks/nope").status_code == 404
    assert client.delete("/tasks/nope").status_code == 404
