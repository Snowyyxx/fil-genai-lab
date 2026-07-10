"""Configuration — reuses the gateway's .env (LangSmith key etc.) if present."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Pull shared secrets from the gateway's .env so we don't duplicate the key.
_gateway_env = Path(__file__).resolve().parent.parent / "llm_gateway" / ".env"
if _gateway_env.exists():
    load_dotenv(_gateway_env)
load_dotenv()  # a local .env, if any, wins

# LiteLLM gateway (OpenAI-compatible) — same proxy the chat app uses.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY") or os.getenv("LITELLM_MASTER_KEY", "sk-local-dev-key")

# Model name as registered in litellm/config.yaml (strong enough for tool-calling).
MODEL = os.getenv("RECO_MODEL", "qwen2.5-3b")

# Trace the agent into its OWN LangSmith project (separate from the chat app).
# We inherit LANGSMITH_API_KEY from the gateway .env, but must *override* its
# LANGSMITH_PROJECT (=llm-gateway) — setdefault wouldn't, since it's already set.
os.environ["LANGSMITH_PROJECT"] = os.getenv("RECO_LANGSMITH_PROJECT", "song-recommender")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
