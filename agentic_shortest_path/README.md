# 🕸️ Agentic Shortest Path — learn agentic AI by watching agents search a graph

A hands-on classroom for **agentic AI**, built with **LangGraph** and an
**open-source** LLM (via Ollama), wrapped in a **Streamlit** UI that teaches as
you run it. The task is deliberately simple — find the cheapest path in a small
weighted graph — so the *agent engineering* stays in focus.

You'll learn, by reading short commented code and running it live:

- **The theory, white-boxed** — a dedicated **📖 Theory** tab teaches the agentic-AI
  foundations end to end (agent vs. workflow, the ReAct loop, tools, planning &
  reflection, memory, multi-agent orchestration, context engineering, the harness,
  evaluation, and when *not* to build an agent) — every idea maps to a pattern you
  can watch run.
- **Colour-coded live runs** — when you run a pattern you don't just see the final
  answer: the **complete graph stays on screen** and lights up as the agent works —
  nodes turn colour as it visits them, edges brighten as it inspects them through
  `get_neighbors`, and its best path is drawn bold. A live **white-box ledger**
  streams every reasoning step (💭 the model's own thinking) and tool call, and a
  scoreboard tracks tool calls, nodes discovered, and best cost vs. the true
  optimum. Nothing is hidden — that's the whole point.
- **LangGraph** — `StateGraph`, state + reducers, nodes, edges, conditional
  edges, `ToolNode`, `Send`, `interrupt`, checkpointers, streaming.
- **Five orchestration patterns**, simple → advanced:
  1. **Single ReAct agent** — the atom of agentic behavior (reason → act → loop).
  2. **Supervisor / orchestrator–worker** — a router delegating to specialist
     agents, with agent-to-agent handoff through shared state.
  3. **Evaluator–optimizer** — a propose → critique → improve reflection loop,
     with **context-window management** you can switch between *trim*, *summarize*,
     and *none*, shown token-by-token.
  4. **Parallel explorers** — map-reduce fan-out with `Send`, fan-in reduce.
  5. **Human-in-the-loop** — the agent pauses via `interrupt()` for your approval;
     a **checkpointer** persists state so it resumes exactly where it stopped.
- **Cross-cutting skills**: agent-to-agent interaction, context-window management,
  memory/persistence, and **harness engineering** (tools, guardrails,
  verification, observability).
- **Note-taking built in**: every pattern has a 📓 Notes panel and a 📋 copyable
  code snippet in the UI; a Deep Dives tab collects snippets for ReAct, A2A,
  context shrinking, memory/HITL, harness engineering, and streaming. Runs stream
  **live**, node by node.

## Setup

```bash
# 1. An open-source, tool-calling model via Ollama
curl -fsSL https://ollama.com/install.sh | sh      # install Ollama
ollama pull qwen2.5:3b-instruct                    # ~2 GB; fast + CPU-friendly (default)
#   (alternatives: llama3.2:3b (fast); qwen2.5:7b-instruct / llama3.1:8b (stronger, ~3× slower on CPU))

# 2. Python deps — use a fresh virtualenv (LangGraph pulls langchain-core 1.x,
#    which can conflict with an older global langchain install)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Launch the classroom
streamlit run app.py
# If you use Anaconda and hit "numpy.core.multiarray failed to import" when a run
# starts, a numpy 2.x in ~/.local is shadowing anaconda's numpy 1.x (its pyarrow is
# built for 1.x). Launch ignoring user site-packages instead:
./run.sh                         # == PYTHONNOUSERSITE=1 streamlit run app.py
```

No GPU required (it runs on CPU). On a laptop CPU a **3B** model is the sweet spot —
a 7B decodes roughly 3× slower per token, which is very noticeable inside an agent
loop. Pick the model from the sidebar dropdown (7B/3B/1.5B presets + custom). Prefer a
different serving stack? Pick **Provider → openai_compatible** in the sidebar and
point it at any local OpenAI-compatible server (vLLM, llama.cpp `--server`,
LM Studio).

## The files (read them in this order)

| File | What it teaches |
|---|---|
| `graph_env.py` | The **environment** — the world agents touch only through tools. Includes Dijkstra as the *answer key* (harness-only ground truth). |
| `llm.py` | Getting an **open-source tool-calling model** behind one uniform interface. |
| `agent_toolkit.py` | **Harness engineering** — tools, the ReAct loop, structured extraction, and context-window trimming. The reusable building blocks. |
| `patterns.py` | The **four LangGraph orchestration patterns**, side by side and heavily commented. |
| `app.py` | The **Streamlit UI** — five tabs: 📖 Theory, 📚 Learn LangGraph, 🕸️ The Graph, 🤖 Run a Pattern (with the colour-coded live exploration view), 🧠 Deep Dives. |

## How the learning is structured

Each pattern in the **Run a Pattern** tab shows: a plain-English explanation, the
concepts it introduces, a **diagram of how the LangGraph is wired**, and — when
you hit Run — a **colour-coded live view** where the complete graph lights up as the
agent explores it, one `get_neighbors` call at a time (side by side with the
orchestration wiring lighting up), a scoreboard, and a **white-box ledger** of every
reasoning step and tool call. The final result is compared against the true optimum, plus
pattern-specific panels (supervisor routing decisions, the reflection loop's
token savings, the parallel branches). The **tool-call log** at the bottom is the
agent's *entire* interaction with the world, so nothing is hidden.

## Why shortest-path?

It's the perfect teaching task for agents: the agent **can't see the whole graph**
(so it must *reason and explore* — genuine agentic behavior), yet every run has a
**checkable ground truth** (Dijkstra), so you always know whether the agent
actually succeeded. Small graphs keep runs fast and cheap while you learn.

> Part of `fil-genai-lab`, alongside `rag_eval/`, `hybrid_reranker/`, and
> `shortest_path/` (the non-agentic combinatorial version — a nice contrast).
