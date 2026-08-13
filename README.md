# fil-genai-lab

A personal lab of GenAI / systems projects. Each top-level folder is its own
self-contained project.

---

## ▶ Current project

### 🧩 [`moe_streaming/`](moe_streaming/) — Giant models, tiny RAM (from scratch)
Building, from zero, the trick behind “run a trillion-parameter model on 8 GB”:
**Mixture-of-Experts sparsity + memory-mapped weight streaming**. Learn-by-building
in Python first, port to C last.
- [`engine/`](moe_streaming/engine/) — **our from-scratch streaming engine**: a REAL Granite MoE whose experts are ripped out to a flat `experts.bin` on disk and streamed in by our own code — top-8-of-32 per layer, per token, straight off the platter. A ~5.8 GB model runs in **under 1 GB of RAM**. It forces + measures real disk reads (major page faults). See [`engine/README.md`](moe_streaming/engine/README.md).
- [`serve.py`](moe_streaming/serve.py) + [`web/chat.html`](moe_streaming/web/chat.html) — **real chat**: pick “🧩 granite-moe (ours)”, type a prompt; the strip shows the **actual routed experts**, and a live panel flips between **● RESIDENT** and **⟳ DISK STREAMING** (toggle “force reads from disk”). Also proxies to Ollama models for comparison.
- [`web/demo.html`](moe_streaming/web/demo.html) — **interactive live simulator**: animated token → router → experts streaming disk→RAM, with the code line highlighting as it runs
- [`web/guide.html`](moe_streaming/web/guide.html) — full illustrated explainer (transformers, MoE, mmap, quantization, C)
- [`examples/phase1_moe_layer.py`](moe_streaming/examples/phase1_moe_layer.py) — runnable demo: 1 GB of experts on disk, ~17 MB in RAM
- [`docs/ROADMAP.md`](moe_streaming/docs/ROADMAP.md) — the 5-phase build plan
- Related reference build: [`kimi_k3_c/`](kimi_k3_c/) (the structured C engine — kept separate)

---

## Other projects

### LLM infra & agents
| Folder | What it is |
|---|---|
| [`llm_gateway/`](llm_gateway/) | Local open-source LLM gateway: FastAPI → **LiteLLM** → Ollama, with Postgres, LangSmith tracing, and an OpenAI-moderation guardrail (Docker Compose). |
| [`song_recommender/`](song_recommender/) | **LangGraph** agentic song recommender — iTunes structured search, cosine-similarity guardrail, likes-based personalization; runs on the gateway. |
| [`agentic_shortest_path/`](agentic_shortest_path/) | Teaching app: learn agentic AI by watching agents search a graph (fog-of-war live view). |
| [`aws-mcp-agent/`](aws-mcp-agent/) | LangGraph + MCP + Amazon Nova/Bedrock agent, deployed on ECS Fargate. |
| [`faraz_shayari/`](faraz_shayari/) | Situation → Ahmad-Faraz-style shayari via RAG (Bedrock/Nova through LiteLLM), on Fargate. |
| [`codingAgent/`](codingAgent/) | Small coding-agent experiment. |
| [`genai_agentic_handbook/`](genai_agentic_handbook/) | Notes / handbook on agentic GenAI. |

### RAG & retrieval
| Folder | What it is |
|---|---|
| [`rag_eval/`](rag_eval/) | RAG evaluation metrics (deterministic retrieval + LLM-as-judge) with an illustrated explainer site and a hands-on notebook. |
| [`hybrid_reranker/`](hybrid_reranker/) | Hybrid retrieval + reranking experiment. |
| [`placements_rag/`](placements_rag/), [`thapar_placement_rag/`](thapar_placement_rag/) | Placement-prep Q&A RAG apps. |

### Systems / algorithms / misc
| Folder | What it is |
|---|---|
| [`kimi_k3_c/`](kimi_k3_c/) | Structured C engine for MoE-from-disk inference (the reference build behind `moe_streaming`). |
| [`bloomfilter/`](bloomfilter/) | Bloom filter implementation. |
| [`shortest_path/`](shortest_path/) | Shortest-path algorithm (non-agentic). |
| [`async_programming/`](async_programming/) | Python async / coroutines practice. |
| [`snowflake/`](snowflake/) | Snowflake data experiment. |
| [`leetcode-categories/`](leetcode-categories/) | LeetCode practice organized by category. |
| [`agentic-ai-deep-dive.html`](agentic-ai-deep-dive.html) | Standalone deep-dive page on agentic AI. |

---

*Layout untouched by design — projects with cross-references (e.g. `song_recommender`
reads `llm_gateway/.env`) stay in place. Large generated artifacts like
`moe_streaming/experts.bin` are git-ignored and rebuilt by their scripts.*
