"""llm.py — one place to get an OPEN-SOURCE chat model that supports tool calling.

Agents = an LLM that can *call tools*. Not every model can; you need one trained
for "function calling". Good open-source picks that run locally via Ollama:

    qwen2.5:3b-instruct   (fast, good tool caller, ~2GB)   <- default (CPU-friendly)
    llama3.2:3b           (fast alternative, ~2GB)
    qwen2.5:7b-instruct   (stronger, ~5GB, ~3× slower on CPU)
    llama3.1:8b           (stronger, ~5GB)

On a CPU (no GPU) prefer a 3B model — a 7B decodes ~3× slower. Set it up once:
    curl -fsSL https://ollama.com/install.sh | sh     # install Ollama
    ollama pull qwen2.5:3b-instruct                   # download the weights (~2GB)
    ollama serve                                       # (usually auto-starts)

We keep TWO backends so you're not locked in:
  * "ollama"            -> langchain-ollama ChatOllama (easiest local option)
  * "openai_compatible" -> any local server that speaks the OpenAI API
                           (vLLM, llama.cpp --server, LM Studio, text-gen-webui)
Both expose the SAME LangChain interface, so the agent code doesn't care which
one you picked. That uniform interface is a big reason people build on LangChain.
"""
from __future__ import annotations

import os


def get_llm(
    provider: str = "ollama",
    model: str = "qwen2.5:3b-instruct",
    base_url: str | None = None,
    temperature: float = 0.0,
):
    """Return a LangChain chat model with `.bind_tools()` support.

    temperature=0 makes runs (mostly) reproducible — good while learning.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )

    raise ValueError(f"unknown provider: {provider!r}")
