"""
LiteLLM the SDK way — call completion() DIRECTLY. No proxy server needed.

This is the style you saw in the video. You import litellm's `completion` and
call it like OpenAI's API, but the `model` string tells LiteLLM which provider
to route to. Here we route straight to the local Ollama daemon.

Contrast with call_litellm.py, which talks to the RUNNING proxy via the openai
client. Same models, but this one needs no server at all.

Run:
    cd llm_gateway
    .venv/bin/python call_litellm_sdk.py
    .venv/bin/python call_litellm_sdk.py "your own prompt"
"""
from __future__ import annotations

import sys

from litellm import completion   # <-- the SDK function the video uses

# model string = "<provider>/<model>". ollama_chat/ applies the chat template.
MODELS = [
    "ollama_chat/llama3:latest",
    "ollama_chat/qwen2.5:3b-instruct",
    "ollama_chat/gemma2:2b",
]
OLLAMA = "http://localhost:11434"   # the local Ollama daemon (no proxy involved)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain what an LLM is in one sentence."
    print(f"Prompt: {prompt}\n")
    for model in MODELS:
        print(f"── {model} " + "─" * max(0, 34 - len(model)))
        resp = completion(                       # <-- called DIRECTLY, no server
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_base=OLLAMA,                     # tell it where Ollama lives
            max_tokens=120,
            temperature=0.7,
        )
        # Response object mirrors OpenAI's shape.
        print(resp.choices[0].message.content.strip(), "\n")


if __name__ == "__main__":
    main()
