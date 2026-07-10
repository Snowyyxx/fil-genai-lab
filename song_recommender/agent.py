"""
The agentic core, as a **LangGraph state machine**.

Control flow is declarative — each decision ("should I search again?", "force a
submit", "fall back") is a graph edge, not a flag buried in a while-loop:

                     ┌───────┐  tool_calls?   ┌───────┐
            START ──►│ agent │───── yes ─────►│ tools │
                     └───┬───┘                └───┬───┘
                         │ no tool_calls          │
              gathered?  │                        │ submitted? ──► END
          ┌──────────────┴──────────────┐         │ else, under step cap ──► agent
          ▼ yes                      no ▼         │ else ──► force_submit
    ┌──────────────┐            ┌──────────┐      │
    │ force_submit │─ no picks ►│ fallback │◄─────┘
    └──────┬───────┘            └────┬─────┘
           │ picks                   │
           ▼                         ▼
          END                       END

Tools:
  - search_songs(query)        → live, STRUCTURED song search (iTunes API).
                                 No local catalog; millions of tracks.
  - submit_recommendations(..) → the FINAL structured answer (ends the graph)

Two safety layers on the output:
  - Grounding — a pick is kept only if it matches a song search_songs returned,
    so the model cannot invent tracks.
  - Guardrail (cosine) — a pick is dropped if its title is too similar to a song
    the user already gave as input (their seed or a previously-liked song). Uses
    a local sentence-transformer, not another LLM call — see `_drop_excluded`.

LangGraph traces to LangSmith automatically (LANGSMITH_TRACING/API_KEY), so no
`wrap_openai` is needed — each node and LLM call shows up in one nested tree.
"""
from __future__ import annotations

import json
import os
import re
from typing import Annotated, Any, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer

from . import likes
from .config import LITELLM_API_KEY, LITELLM_BASE_URL, MODEL
from .search import search_songs

MAX_STEPS = 4

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_songs",
            "description": "Search the live music database (iTunes) for real songs. Returns a list of songs with title, artist, genre and year. Call this to get songs you can recommend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Artist, genre, mood or vibe, e.g. 'Nusrat Fateh Ali Khan qawwali' or 'punjabi party Diljit'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_recommendations",
            "description": "Submit the FINAL picks. Use ONLY song titles returned by search_songs. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Song title, exactly as returned by search_songs."},
                                "reason": {"type": "string", "description": "One short sentence: why it fits the user."},
                            },
                            "required": ["title", "reason"],
                        },
                    }
                },
                "required": ["recommendations"],
            },
        },
    },
]

SYSTEM = (
    "You are a music recommendation agent. Recommend real songs matching the user's "
    "language, genre/type, mood, and taste.\n"
    "Process: (1) call search_songs to get real candidate songs; (2) call "
    "submit_recommendations picking the best ones. Usually one search is enough.\n"
    "Rules: ONLY pick titles that search_songs returned — never invent a song. "
    "Prefer variety (mix artists). Keep each reason to one short, specific sentence."
)

_llm = ChatOpenAI(
    model=MODEL, base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY,
    temperature=0.4, max_tokens=500,
)
_llm_tools = _llm.bind_tools(TOOLS)

# ── Cosine-similarity guardrail ─────────────────────────────────────────────
# A small sentence-transformer embeds song titles; a recommendation is dropped
# if its title is too close (cosine) to any song the user already gave as input.
# Catches remixes / "(From ...)" variants / spelling diffs that exact matching
# misses — no extra LLM call, just a fast local embedding.
_embedder = SentenceTransformer(os.getenv("RECO_EMBED_MODEL", "all-MiniLM-L6-v2"))
SIM_THRESHOLD = float(os.getenv("RECO_GUARDRAIL_SIM", "0.82"))


def _norm_title(t: str) -> str:
    """Lowercase and strip parenthetical suffixes like '(From \"...\")' / '(Lofi Flip)'."""
    return re.sub(r"\s*\(.*?\)\s*", " ", (t or "").lower()).strip()


def _seed_phrases(seed: str) -> list[str]:
    """Song-name phrases the user typed, split out of the free-text seed."""
    parts = re.split(r"[,/&]|\bby\b|\band\b|\bfeat\.?\b", (seed or "").lower())
    return [p for p in (_norm_title(p) for p in parts) if len(p) >= 4]


def _drop_excluded(songs: list[dict], exclude: list[str]) -> list[dict]:
    """GUARDRAIL: remove songs whose title is cosine-similar (>= threshold) to
    any of the user's input songs. Batched — one embed call per side."""
    if not exclude or not songs:
        return songs
    anchors = _embedder.encode(list(exclude), normalize_embeddings=True)          # (m, 384)
    vecs = _embedder.encode([_norm_title(s["title"]) for s in songs],
                            normalize_embeddings=True)                            # (n, 384)
    sims = vecs @ anchors.T                                                       # (n, m) cosine
    return [s for i, s in enumerate(songs) if float(np.max(sims[i])) < SIM_THRESHOLD]


# ── Grounding + guardrail: only songs that search_songs returned, and never a
#    song too similar to a user input, may be recommended. ─────────────────────
def _finalize(recs: list[dict], candidates: dict[str, dict],
              exclude: list[str]) -> list[dict]:
    resolved, seen = [], set()
    for r in recs:
        title = (r.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        song = candidates.get(key)
        if song is None:  # tolerate near-misses (e.g. trailing "(From ...)")
            song = next((v for k, v in candidates.items()
                         if key in k or k in key), None)
        if song is None or song["title"].lower() in seen:
            continue                       # hallucinated / not in candidates → drop
        seen.add(song["title"].lower())
        resolved.append({
            "title": song["title"], "artist": song["artist"],
            "genre": song.get("genre", ""), "year": song.get("year", ""),
            "url": song.get("url"), "reason": r.get("reason", ""),
        })
    return _drop_excluded(resolved, exclude)   # GUARDRAIL (cosine)


def _fill_from_candidates(candidates: dict[str, dict], exclude: list[str], count: int) -> list[dict]:
    """When the model's picks are all invalid/guardrail-excluded, fill from the
    candidate pool it already fetched — deterministic, avoids a slow resubmit loop."""
    songs = _drop_excluded(list(candidates.values()), exclude)
    return [{"title": s["title"], "artist": s["artist"], "genre": s.get("genre", ""),
             "year": s.get("year", ""), "url": s.get("url"),
             "reason": "Matches your taste."} for s in songs[:count]]


def _fallback_recs(query: str, count: int, exclude: list[str]) -> list[dict]:
    """Degraded but still real songs — straight from the music search, with the
    same cosine guardrail applied."""
    songs = [{**s, "reason": "Top match from music search."}
             for s in search_songs(query, limit=count + 5).get("songs", [])]
    return _drop_excluded(songs, exclude)[:count]


# ── Graph state ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    count: int
    query_hint: str
    steps: list[str]
    candidates: dict[str, dict]   # lowercase title → song (the grounding set)
    exclude: list[str]            # guardrail: song-name phrases never to recommend
    recommendations: list[dict]
    mode: str
    gathered: bool
    iterations: int


# ── Nodes ───────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    resp = _llm_tools.invoke(state["messages"])
    return {"messages": [resp], "iterations": state["iterations"] + 1}


def tools_node(state: AgentState) -> dict:
    last: AIMessage = state["messages"][-1]
    msgs, steps = [], list(state["steps"])
    gathered, recs, mode = state["gathered"], state["recommendations"], state["mode"]
    candidates = dict(state["candidates"])

    for tc in last.tool_calls:
        name, args = tc["name"], (tc["args"] or {})
        steps.append(name)

        if name == "search_songs":
            result = search_songs(args.get("query", ""), limit=12)
            for s in result.get("songs", []):
                candidates[s["title"].lower()] = s
            gathered = True
            slim = [{"title": s["title"], "artist": s["artist"], "genre": s["genre"], "year": s["year"]}
                    for s in result.get("songs", [])]
            content = json.dumps({"songs": slim})[:1800]
        elif name == "submit_recommendations":
            picked = _finalize(args.get("recommendations", []), candidates, state["exclude"])
            if not picked and candidates:
                # all picks invalid or dropped by the guardrail → fill from the
                # candidate pool rather than trigger an expensive resubmit loop
                picked = _fill_from_candidates(candidates, state["exclude"], state["count"])
            if picked:
                recs, mode = picked, "agent"
                content = "ok"
            else:
                content = json.dumps({"error": "search returned nothing usable; "
                                               "call search_songs with a broader query"})
        else:
            content = json.dumps({"error": f"unknown tool {name}"})

        msgs.append(ToolMessage(content=content, tool_call_id=tc["id"], name=name))

    return {"messages": msgs, "steps": steps, "gathered": gathered, "candidates": candidates,
            "recommendations": recs, "mode": mode}


def force_submit_node(state: AgentState) -> dict:
    """Model searched but never submitted → compel the final answer.

    Two attempts, because Ollama often *ignores* a forced `tool_choice`:
      1. force the submit_recommendations tool call
      2. plain JSON extraction (reliable with small local models)
    """
    steps = state["steps"] + ["submit_recommendations*"]
    cands = state["candidates"]
    excl = state["exclude"]

    try:
        forced = _llm.bind_tools(TOOLS, tool_choice="submit_recommendations")
        resp: AIMessage = forced.invoke(state["messages"] + [HumanMessage(
            content="Now call submit_recommendations, picking the best songs from the "
                    "search_songs results above. Prefer variety."
        )])
        for tc in (resp.tool_calls or []):
            if tc["name"] == "submit_recommendations":
                picked = _finalize((tc["args"] or {}).get("recommendations", []), cands, excl)
                if picked:
                    return {"recommendations": picked, "mode": "agent", "steps": steps}
    except Exception:
        pass

    try:
        listing = "\n".join(f"- {s['title']} — {s['artist']}" for s in list(cands.values())[:12])
        resp = _llm.invoke(state["messages"][:2] + [HumanMessage(
            content=f"Choose {state['count']} songs from EXACTLY this list:\n{listing}\n\n"
                    f"Respond with ONLY a JSON array, no prose, no markdown:\n"
                    f'[{{"title": "exact title from the list", "reason": "one short sentence"}}]'
        )])
        match = re.search(r"\[.*\]", str(resp.content), re.S)
        if match:
            picked = _finalize(json.loads(match.group(0)), cands, excl)
            if picked:
                return {"recommendations": picked, "mode": "agent",
                        "steps": steps + ["json_extract"]}
    except Exception:
        pass

    return {"steps": steps}


def fallback_node(state: AgentState) -> dict:
    return {"recommendations": _fallback_recs(state["query_hint"], state["count"], state["exclude"]),
            "mode": "fallback"}


# ── Edges (the control flow that used to be flags + try/except) ─────────────
def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "force_submit" if state["gathered"] else "fallback"


def route_after_tools(state: AgentState) -> str:
    if state["recommendations"]:
        return END
    if state["iterations"] >= MAX_STEPS:
        return "force_submit" if state["gathered"] else "fallback"
    return "agent"


def route_after_force(state: AgentState) -> str:
    return END if state["recommendations"] else "fallback"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("force_submit", force_submit_node)
    g.add_node("fallback", fallback_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "force_submit": "force_submit", "fallback": "fallback"})
    g.add_conditional_edges("tools", route_after_tools,
                            {"agent": "agent", "force_submit": "force_submit",
                             "fallback": "fallback", END: END})
    g.add_conditional_edges("force_submit", route_after_force,
                            {END: END, "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


GRAPH = _build_graph()


# ── Public API (unchanged contract) ─────────────────────────────────────────
def recommend(
    user_id: str,
    language: str = "",
    genre: str = "",
    mood: str = "",
    seed: str = "",
    count: int = 8,
) -> dict[str, Any]:
    profile = likes.get_taste_profile(user_id)
    liked_titles = [s["title"] for s in profile.get("liked_songs", []) if s.get("title")]
    if profile.get("liked_songs"):
        liked = ", ".join(f"{s['title']} ({s['artist']})" if s.get("artist") else s["title"]
                          for s in profile["liked_songs"][:6])
        taste_line = f"My past likes: {liked}. Favor SIMILAR songs, but do NOT recommend these exact ones."
    else:
        taste_line = "No listening history yet."

    # GUARDRAIL inputs: never recommend a song the user already gave —
    # both previously-liked songs and songs named in the seed text.
    exclude = sorted({_norm_title(t) for t in liked_titles} | set(_seed_phrases(seed)))

    query_hint = " ".join(x for x in [seed, genre, language, mood] if x) or "popular songs"
    user_prompt = (
        f"Recommend {count} songs.\n"
        f"Language: {language or 'any'}\nType/Genre: {genre or 'any'}\n"
        f"Mood: {mood or 'any'}\nI currently like: {seed or '(not specified)'}\n"
        f"{taste_line}\n"
        f"IMPORTANT: do NOT recommend any song I already mentioned above "
        f"(my current likes or seed) — suggest NEW songs only.\n"
        f"Call search_songs with a query like \"{query_hint}\", then submit the best picks."
    )

    init: AgentState = {
        "messages": [SystemMessage(content=SYSTEM), HumanMessage(content=user_prompt)],
        "count": count, "query_hint": query_hint, "steps": [], "candidates": {},
        "exclude": exclude,
        "recommendations": [], "mode": "fallback", "gathered": False, "iterations": 0,
    }

    final = GRAPH.invoke(init, config={"recursion_limit": 25, "run_name": "recommend_agent"})
    return {
        "recommendations": final["recommendations"][:count],
        "steps": final["steps"],
        "mode": final["mode"],
    }
