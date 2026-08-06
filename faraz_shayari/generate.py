"""
generate.py — the G in RAG. Retrieve Faraz couplets close to the situation, then
have the LiteLLM GATEWAY (→ Bedrock Nova) compose a NEW shayari in his voice.

The app now speaks ONLY to the gateway (OpenAI protocol); the gateway holds the
Bedrock creds and does the routing. So this container needs no AWS keys.
"""
from __future__ import annotations

import os

from retrieve import retrieve, GATEWAY

GEN_MODEL = os.getenv("GEN_MODEL", "nova-pro")     # a model_name in the gateway config

SYSTEM = (
    "You are a poet who writes original ghazal-style shayari in the voice of the "
    "legendary Urdu poet Ahmad Faraz. You write in ROMAN URDU (Latin script with "
    "diacritics like a i u n kh gh), never in Urdu or Devanagari script. His hallmarks: "
    "love and separation (hijr), longing, wounded dignity, gentle defiance, and a soft "
    "melancholy. You compose a NEW couplet or two — never copy the sample couplets, only "
    "echo their tone, diction, and rhythm (radif/qaafiya feel)."
)


def build_prompt(situation: str, couplets: list[dict]) -> str:
    examples = "\n\n".join(f"{c['line1']}\n{c['line2']}" for c in couplets)
    return (
        f"The user's situation:\n\"{situation}\"\n\n"
        f"Ahmad Faraz's real couplets on kindred themes (for STYLE ONLY - do not copy):\n"
        f"{examples}\n\n"
        f"Now compose an original Roman-Urdu shayari (2-4 lines) that speaks to this "
        f"situation in Faraz's voice. Output the shayari, then on a new line a one-line "
        f"English meaning prefixed with 'meaning: '."
    )


def make_shayari(situation: str, k: int = 4) -> dict:
    couplets = retrieve(situation, k=k)
    resp = GATEWAY.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(situation, couplets)}],
        max_tokens=300, temperature=0.8,
    )
    return {
        "situation": situation,
        "shayari": resp.choices[0].message.content.strip(),
        "inspirations": [
            {"couplet": f"{c['line1']} / {c['line2']}", "ghazal": c["ghazal"],
             "score": round(c["score"], 3)}
            for c in couplets
        ],
    }


if __name__ == "__main__":
    import sys
    out = make_shayari(sys.argv[1] if len(sys.argv) > 1 else "I miss someone who left me")
    print(out["shayari"], "\n\ninspired by:")
    for i in out["inspirations"]:
        print(" ", i["couplet"])
