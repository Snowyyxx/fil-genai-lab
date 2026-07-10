"""
Song search — the agent's grounding source (there is no local catalog).

Queries the **iTunes Search API**: keyless, free, and covers millions of live
tracks. Returns STRUCTURED songs ({title, artist, genre, year, url}), so the LLM
only has to *pick* songs, not extract them from prose — which small local models
do far more reliably.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from langsmith import traceable

_ITUNES = "https://itunes.apple.com/search"
_UA = {"User-Agent": "song-recommender/0.3"}


@traceable(run_type="tool", name="search_songs")
def search_songs(query: str, limit: int = 12, country: str = "US") -> dict[str, Any]:
    """Structured song search over the live iTunes music database.

    `@traceable` makes this call a child span of the LangGraph agent run in
    LangSmith, so you see the exact query and results inside the trace tree.
    """
    params = urllib.parse.urlencode(
        {"term": query, "entity": "song", "limit": max(3, min(limit, 25)), "country": country}
    )
    req = urllib.request.Request(f"{_ITUNES}?{params}", headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                      # network / API failure → empty (agent falls back)
        return {"songs": [], "error": str(e)}

    songs = []
    for x in data.get("results", []):
        if not x.get("trackName"):
            continue
        songs.append({
            "title": x["trackName"],
            "artist": x.get("artistName", ""),
            "genre": x.get("primaryGenreName", ""),
            "year": (x.get("releaseDate") or "")[:4],
            "url": x.get("trackViewUrl"),         # iTunes/Apple Music preview link
        })
    return {"songs": songs}
