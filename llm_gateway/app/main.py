"""
A simple FastAPI app that talks to open-source models through the LiteLLM proxy.

Because LiteLLM exposes an OpenAI-compatible API, we just point the standard
`openai` client at it — no special SDK needed. The same code would work against
OpenAI, Anthropic, Bedrock, etc. if you re-pointed the proxy; here everything is
local and open-source (Ollama behind LiteLLM).

Endpoints
---------
GET  /health   liveness + which models the proxy currently serves
GET  /models   list model names available through the gateway
POST /chat     send a message, get a completion back
POST /chat/stream  same, but streams tokens as they're generated
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI, APIConnectionError, APIStatusError
from pydantic import BaseModel, Field

# ── Config (all overridable via environment) ────────────────────────────────
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-local-dev-key")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")

# The OpenAI SDK pointed at the LiteLLM proxy.
client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)

app = FastAPI(title="Open-Source LLM Gateway", version="0.1.0")

# Serve the single-page chat UI (app/static/index.html) at "/".
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Request / response schemas ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message.")
    model: str = Field(DEFAULT_MODEL, description="Model name as defined in LiteLLM config.")
    system: str | None = Field(None, description="Optional system prompt.")
    history: list[dict] = Field(
        default_factory=list,
        description="Prior turns for multi-turn chat, e.g. [{'role':'user','content':'...'}, ...].",
    )
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=8192)

class ChatResponse(BaseModel):
    model: str
    reply: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _build_messages(req: ChatRequest) -> list[dict]:
    messages: list[dict] = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.extend(req.history)  # prior turns, oldest first
    messages.append({"role": "user", "content": req.message})
    return messages


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Confirms the proxy is reachable and reports the served models."""
    try:
        models = client.models.list()
        return {
            "status": "ok",
            "litellm": LITELLM_BASE_URL,
            "models": [m.id for m in models.data],
        }
    except Exception as e:  # connection refused, auth, etc.
        raise HTTPException(status_code=503, detail=f"LiteLLM unreachable: {e}")


@app.get("/models")
def list_models():
    try:
        return {"models": [m.id for m in client.models.list().data]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """One-shot chat completion."""
    try:
        resp = client.chat.completions.create(
            model=req.model,
            messages=_build_messages(req),
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"cannot reach LiteLLM: {e}")
    except APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    usage = resp.usage
    return ChatResponse(
        model=resp.model,
        reply=resp.choices[0].message.content or "",
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Streams the reply token-by-token as text/plain."""

    def token_generator():
        try:
            stream = client.chat.completions.create(
                model=req.model,
                messages=_build_messages(req),
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta 
        except (APIConnectionError, APIStatusError) as e:
            yield f"\n[error: {e}]"

    return StreamingResponse(token_generator(), media_type="text/plain")
