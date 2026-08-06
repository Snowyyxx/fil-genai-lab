"""
retrieve.py — the R in RAG. Embed the user's situation and find the most
similar real Faraz couplets by cosine similarity.

The embedding now goes through the LiteLLM GATEWAY (OpenAI-compatible), which
forwards to Bedrock Titan. So this process needs NO AWS creds — only the gateway
does. (Verified: gateway Titan embeddings == the boto3 ones used to build the
index, cosine 1.0.)

Try it:
    LITELLM_BASE_URL=http://localhost:4001/v1 .venv/bin/python retrieve.py "I miss them"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

HERE = Path(__file__).parent
GATEWAY = OpenAI(
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4001/v1"),
    api_key=os.getenv("LITELLM_API_KEY", "sk-bedrock-gateway"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "titan-embed")

_rows = [json.loads(l) for l in (HERE / "data" / "couplets.jsonl").read_text("utf-8").splitlines()]
_emb = np.load(HERE / "data" / "embeddings.npy")     # [N, 1024], unit-normalized


def embed_query(text: str) -> np.ndarray:
    r = GATEWAY.embeddings.create(model=EMBED_MODEL, input=text)
    v = np.asarray(r.data[0].embedding, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def retrieve(situation: str, k: int = 5) -> list[dict]:
    q = embed_query(situation)
    scores = _emb @ q                       # cosine sim (all unit vectors)
    top = np.argsort(-scores)[:k]
    return [{**_rows[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    situation = sys.argv[1] if len(sys.argv) > 1 else "I miss someone who left me"
    print(f"Situation: {situation}\n")
    for r in retrieve(situation, k=5):
        print(f"[{r['score']:.3f}]  ({r['ghazal'][:36]}…)")
        print(f"   {r['line1']}\n   {r['line2']}\n")
