"""
Lightweight taste/feedback store — a JSON file, no database.

The catalog DB is gone, so likes are keyed by (title, artist) text. This keeps
personalization (liked songs shape the next prompt) without any DB dependency.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

_LOCK = threading.Lock()
_PATH = os.path.join(os.path.dirname(__file__), "likes.json")


def _load() -> dict:
    if os.path.exists(_PATH):
        try:
            with open(_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save(d: dict) -> None:
    with open(_PATH, "w") as f:
        json.dump(d, f, indent=1)


def get_taste_profile(user_id: str) -> dict[str, Any]:
    likes = _load().get(user_id, [])
    return {"liked_songs": likes, "likes_count": len(likes)}


def record_feedback(user_id: str, title: str, artist: str, liked: bool) -> dict[str, Any]:
    key = (title.strip().lower(), (artist or "").strip().lower())
    with _LOCK:
        d = _load()
        likes = [l for l in d.get(user_id, [])
                 if (l["title"].lower(), (l.get("artist") or "").lower()) != key]
        if liked:
            likes.append({"title": title.strip(), "artist": (artist or "").strip()})
        d[user_id] = likes
        _save(d)
    return get_taste_profile(user_id)
