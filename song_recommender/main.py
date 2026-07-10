"""FastAPI app for the agentic (web-search) song recommender."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import likes
from .agent import recommend

app = FastAPI(title="Agentic Song Recommender (web search)", version="0.2.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class RecommendRequest(BaseModel):
    user_id: str = Field("demo", description="Who we're recommending for.")
    language: str = ""
    genre: str = Field("", description="Type: Qawwali, Bollywood, Sufi, Pop, …")
    mood: str = ""
    seed: str = Field("", description="Current taste, e.g. 'Nusrat Fateh Ali Khan, Arijit'.")
    count: int = Field(8, ge=1, le=15)


class FeedbackRequest(BaseModel):
    user_id: str = "demo"
    title: str
    artist: str = ""
    liked: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "retrieval": "itunes_search"}


@app.post("/recommend")
def do_recommend(req: RecommendRequest):
    try:
        return recommend(
            user_id=req.user_id, language=req.language, genre=req.genre,
            mood=req.mood, seed=req.seed, count=req.count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def do_feedback(req: FeedbackRequest):
    return {"ok": True, "profile": likes.record_feedback(req.user_id, req.title, req.artist, req.liked)}


@app.get("/profile")
def profile(user_id: str = "demo"):
    return likes.get_taste_profile(user_id)
