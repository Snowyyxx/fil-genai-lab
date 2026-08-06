"""
Call the LiteLLM endpoint (3 open-source LLMs) from Python.

LiteLLM speaks the OpenAI protocol, so you use the normal `openai` client and
just point base_url at the proxy. Switching models = changing one string.

Run:
    cd llm_gateway
    .venv/bin/pip install openai        # if not already installed
    .venv/bin/python call_litellm.py
    .venv/bin/python call_litellm.py "your own prompt here"
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI

# Point the OpenAI client at the LiteLLM proxy instead of api.openai.com.
client = OpenAI(
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    api_key=os.getenv("LITELLM_MASTER_KEY", "sk-local-dev-key"),  # the proxy's Bearer key
)

# The three model_names from config.opensource.yaml.
MODELS = ["llama3", "qwen2.5", "gemma2"]


def ask(model: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,                                   # <- just a string; LiteLLM routes it
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain what an LLM is in one sentence."
    print(f"Prompt: {prompt}\n")
    for model in MODELS:
        print(f"── {model} " + "─" * (40 - len(model)))
        try:
            print(ask(model, prompt), "\n")
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
