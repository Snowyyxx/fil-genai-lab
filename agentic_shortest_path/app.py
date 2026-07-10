"""app.py — a Streamlit classroom for agentic AI, taught through shortest-path.

Run it:
    ollama pull qwen2.5:3b-instruct        # one-time: a fast, CPU-friendly tool caller
    pip install -r requirements.txt
    streamlit run app.py

Tabs:
  📖 Theory           — the agentic-AI foundations, self-contained (agent vs.
                          workflow, ReAct, tools, planning, memory, orchestration,
                          context engineering, the harness, evaluation).
  📚 Learn LangGraph  — the framework's core ideas, with copyable code.
  🕸️ The Graph        — the problem world + the ground-truth answer.
  🤖 Run a Pattern     — pick a pattern, read the notes + code, run it LIVE and
                          watch the agent explore the graph, colour-coded by progress.
  🧠 Deep Dives        — agentic reasoning, A2A, context shrinking, harness eng.,
                          plus a copy-paste snippet library.

Every pattern in "Run a Pattern" has a 📓 Notes panel and a 📋 Code panel so you
can take notes and copy the code into your own notebook. Streamlit's code blocks
have a built-in copy button. Running a pattern streams a **colour-coded exploration**
of the complete graph — nodes and edges light up as the agent visits/inspects them —
plus a white-box ledger of every reasoning step and tool call, so nothing is hidden.
"""
from __future__ import annotations

import re
import time

import streamlit as st

from graph_env import random_graph

st.set_page_config(page_title="Agentic Shortest Path", layout="wide", page_icon="🕸️")

try:
    from llm import get_llm
    from patterns import PATTERNS
    from langgraph.types import Command
    IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    PATTERNS, Command, IMPORT_ERROR = {}, None, e


# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #
MODEL_PRESETS = [
    "qwen2.5:3b-instruct",    # fast, good tool caller — best default on a CPU laptop
    "llama3.2:3b",            # fast alternative
    "qwen2.5:1.5b-instruct",  # fastest; weaker at tools + the DFS bookkeeping
    "qwen2.5:7b-instruct",    # stronger reasoning, but ~3× slower on CPU
    "llama3.1:8b",            # stronger; slower
    "custom…",
]

with st.sidebar:
    st.header("⚙️ Model")
    provider = st.selectbox("Provider", ["ollama", "openai_compatible"], index=0)
    model_choice = st.selectbox(
        "Model", MODEL_PRESETS, index=0,
        help="Smaller = faster on CPU. 3B is the sweet spot for a laptop; 7B/8B reason "
             "better but decode ~3× slower. Pick 'custom…' to type any model id.")
    model = (st.text_input("Custom model id", value="qwen2.5:3b-instruct")
             if model_choice == "custom…" else model_choice)
    st.caption("⚡ Smaller models decode faster on CPU. Close other RAM-hungry apps too.")
    base_url = st.text_input("Base URL (blank = default)", value="")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    st.divider()
    st.header("🕸️ Graph")
    n = st.slider("Nodes", 4, 9, 6)
    density = st.slider("Edge density", 0.2, 1.0, 0.5, 0.1)
    seed = st.number_input("Seed", value=7, step=1)

graph_key = (n, density, seed)
if st.session_state.get("graph_key") != graph_key:
    st.session_state.env = random_graph(n=n, density=density, seed=int(seed))
    st.session_state.graph_key = graph_key
    st.session_state.pop("hitl", None)  # reset any paused HITL run
env = st.session_state.env


def make_llm():
    return get_llm(provider=provider, model=model,
                   base_url=base_url or None, temperature=temperature)


def problem_dot(env, highlight: list | None = None) -> str:
    hp = set(zip(highlight, highlight[1:])) if highlight else set()
    out = ["digraph { rankdir=LR; node [shape=circle, fontname=Helvetica];"]
    out.append(f'{env.source} [style=filled, fillcolor="#cfe8ff"];')
    out.append(f'{env.target} [style=filled, fillcolor="#ffd7d7"];')
    for u in env.nodes():
        for v, w in env.neighbors(u):
            hot = (u, v) in hp
            color = '"#d62728"' if hot else "gray40"
            pw = 2.6 if hot else 1
            out.append(f'{u} -> {v} [label="{int(w)}", color={color}, penwidth={pw}];')
    out.append("}")
    return "\n".join(out)


st.title("🕸️ Agentic Shortest Path")
st.caption("Learn agentic AI — LangGraph, multi-agent orchestration, agent-to-agent "
           "handoffs, context-window management, memory & human-in-the-loop, and "
           "harness engineering — by watching open-source LLM agents search a graph.")

if IMPORT_ERROR is not None:
    st.error("LangGraph isn't installed, so the agent patterns can't load.\n\n"
             f"```\n{IMPORT_ERROR}\n```\n\nInstall it:\n```\npip install -r requirements.txt\n```")

tab_theory, tab_learn, tab_graph, tab_run, tab_deep = st.tabs(
    ["📖 Theory", "📚 Learn LangGraph", "🕸️ The Graph", "🤖 Run a Pattern", "🧠 Deep Dives"])


# =========================================================================== #
# TAB: Theory — the agentic-AI foundations, self-contained.                   #
# =========================================================================== #
with tab_theory:
    st.markdown(
        "## The theory, white-boxed\n"
        "Everything you need to understand *why* the patterns in this app are built "
        "the way they are — from the definition of an 'agent' to the harness that "
        "makes one dependable. Each idea maps to something you can **watch run** in "
        "the **🤖 Run a Pattern** tab. Read top-to-bottom, or jump to a concept.")
    st.info("**The one-sentence version:** an *agent* is an LLM put in a **loop** with "
            "**tools** and a **stopping condition**, wrapped in a **harness** that "
            "keeps it honest, bounded, and observable. Everything below is a "
            "consequence of that sentence.")

    with st.expander("1 · Agent vs. workflow — the spectrum (start here)", expanded=True):
        st.markdown(
            "The most common early mistake is calling everything an 'agent'. There is "
            "a useful line:\n\n"
            "- **Workflow** — LLMs and tools orchestrated through **code paths you "
            "wrote**: *summarize → classify → route*. The model fills in steps, but "
            "*you* fix the sequence. Predictable, testable, cheap.\n"
            "- **Agent** — the LLM **dynamically directs its own process**: it decides "
            "which tool to call, in what order, how many times, and when it's done. "
            "You trade predictability for flexibility on problems you can't fully "
            "script in advance.\n\n"
            "The practical consequence: a *workflow's* cost is bounded by its code; an "
            "*agent's* cost is bounded only by its **stopping condition** — which is "
            "exactly why guardrails (max steps, budgets) are not optional.")
        st.markdown(
            "| Dimension | Workflow | Agent |\n"
            "|---|---|---|\n"
            "| Control flow | Fixed in your code | Chosen by the model at runtime |\n"
            "| Predictability | High — same path every run | Low — varies per input |\n"
            "| Best for | Well-defined, decomposable tasks | Open-ended, unknown # of steps |\n"
            "| Cost / latency | Bounded, estimable | Variable — needs guardrails |\n"
            "| Failure mode | Wrong output at a known step | Loops, tool thrash, runaway cost |")
        st.caption("'Agentic' is a spectrum, not a binary. Most good production systems "
                   "are **workflows with a small agentic core** — this app's supervisor "
                   "pattern (LLM-proposes / harness-disposes) is exactly that shape.")

    with st.expander("2 · The agent loop — ReAct (reason → act → observe → repeat)"):
        st.markdown(
            "The dominant agent pattern is **ReAct = Reason + Act**. The model "
            "interleaves *reasoning* ('what do I know, what next?') with *actions* "
            "(tool calls), then *observes* the result and folds it back into context. "
            "The loop continues until the model answers **without** requesting a tool.\n\n"
            "- **Reason** — think about the goal and current state.\n"
            "- **Act** — request a tool call with structured arguments.\n"
            "- **Observe** — the harness runs the tool and appends the result.\n"
            "- **Repeat** — the model sees the observation and either acts again or stops.\n\n"
            "The loop is **stateless on the model side**: the agent's entire 'memory' "
            "of what it's done is the *growing transcript* you resend each turn. That "
            "single fact is why context management (idea 7) matters so much.")
        st.code('''def run_react(model, tools, messages, max_steps=8):   # this app's agent_toolkit.py
    convo = list(messages)
    for _ in range(max_steps):                 # GUARDRAIL: never an unbounded loop
        ai = model.invoke(convo); convo.append(ai)     # REASON (+ maybe request a tool)
        if not ai.tool_calls:                  # no tool -> the agent is done
            break
        for call in ai.tool_calls:             # ACT
            result = run_tool(call)            # OBSERVE
            convo.append(ToolMessage(result, tool_call_id=call["id"]))  # REPEAT
    return convo''', language="python")
        st.caption("▶ Watch it: **Pattern 1 · Single ReAct Agent**. In the live "
                   "exploration view you see reason→act→observe light up the graph, "
                   "one `get_neighbors` at a time.")

    with st.expander("3 · Tools & function calling — the harness's hands"):
        st.markdown(
            "Tools are how an agent reaches beyond its own context — to search, query, "
            "call an API, run code, or (here) *sense the graph*. A tool is three "
            "things: a **name** the model calls, a **description** that teaches it "
            "*when* to use the tool, and a **typed schema** for the arguments.\n\n"
            "The **description is the most important field** — the model routes almost "
            "entirely on it. Be prescriptive about the *trigger condition*, not just "
            "the mechanics. In this app, `submit_path`'s docstring literally says "
            "*'ALWAYS finish by calling this — never state a cost you computed "
            "yourself'*, which is how we stop the model from trusting its own "
            "arithmetic.")
        st.code('''@tool
def get_neighbors(node: str) -> str:
    """Look up the outgoing edges from `node` as 'neighbor:cost' pairs.
    This is how you explore the graph one step at a time."""     # <- the model reads THIS
    ...

@tool
def submit_path(path: str) -> str:
    """Submit a full route 'A,C,E'. Returns validity + the EXACT cost.
    ALWAYS finish by calling this — never state a cost you computed yourself."""
    ...''', language="python")
        st.caption("Two tools = two senses. The agent can *look* (`get_neighbors`) and "
                   "*commit* (`submit_path`). Everything it knows, it learned by calling them.")

    with st.expander("4 · Planning & reflection — thinking before and after acting"):
        st.markdown(
            "**Planning** is reasoning about *multiple* steps before committing — "
            "'a cheap first edge can lead to an expensive remainder, so compare a "
            "couple of options first.' **Reflection** (a.k.a. self-critique) is the "
            "agent evaluating its *own* output and trying again.\n\n"
            "Reflection is what lifts an agent above one-shot prompting, but a model "
            "grading itself can loop forever or declare victory too early. So a good "
            "reflection loop has **two independent stop signals**:\n"
            "1. **Ground truth** — an objective check (here: *is the cost optimal?*).\n"
            "2. **A loop guard** — a hard `max_iterations` cap, because you never "
            "trust a model to always converge.")
        st.caption("▶ Watch it: **Pattern 3 · Evaluator–Optimizer**. The optimizer "
                   "proposes, the evaluator critiques and decides stop-or-retry.")

    with st.expander("5 · Memory & persistence — the taxonomy"):
        st.markdown(
            "'Memory' is an overloaded word. Three distinct kinds:\n\n"
            "- **Working memory (the context window).** The current transcript — "
            "messages, tool results, scratch reasoning. Fast but finite; managed by "
            "trimming/summarizing (idea 7). In LangGraph this is the `add_messages` "
            "channel.\n"
            "- **Persistence (the checkpointer).** LangGraph's `MemorySaver` (or a "
            "SQLite/Postgres saver) snapshots the *whole graph state* at every step, "
            "keyed by a `thread_id`. This one feature unlocks **resume-after-crash**, "
            "**time-travel**, and **human-in-the-loop** pausing.\n"
            "- **Long-term memory (cross-session).** Facts written to a store/vector "
            "DB and retrieved in *future* runs — beyond this app's scope, but the same "
            "idea: state that outlives the loop.")
        st.code('''from langgraph.checkpoint.memory import MemorySaver
graph  = g.compile(checkpointer=MemorySaver())        # persistence -> resume/HITL
config = {"configurable": {"thread_id": "run-1"}}     # the key the state is stored under''',
                language="python")
        st.caption("▶ Watch it: **Pattern 5 · Human-in-the-Loop** — the checkpointer is "
                   "what lets the run *pause* at `interrupt()` and resume exactly there.")

    with st.expander("6 · Multi-agent orchestration — the spectrum, and this app's 5 patterns"):
        st.markdown(
            "One agent isn't always enough. As tasks grow you climb a ladder of "
            "orchestration — but **only climb when the task forces you to**; every "
            "rung adds tokens, latency, and failure modes.")
        st.markdown(
            "| Pattern | Shape | What it buys you | In this app |\n"
            "|---|---|---|---|\n"
            "| Single agent | one ReAct loop | the baseline atom | **Pattern 1** |\n"
            "| Prompt chaining | fixed A→B→C | decompose a known pipeline | (workflow) |\n"
            "| Routing / dispatch | classify → handler | send work to a specialist | supervisor core |\n"
            "| Parallelization | fan out, vote/merge | speed + ensembling | **Pattern 4** |\n"
            "| Orchestrator–worker | supervisor delegates | dynamic delegation | **Pattern 2** |\n"
            "| Evaluator–optimizer | propose → critique → retry | self-correction | **Pattern 3** |\n"
            "| Human-in-the-loop | pause for approval | control over risk | **Pattern 5** |")
        st.markdown(
            "**How agents talk to each other (A2A).** Two mechanisms:\n"
            "- **Shared-state channels** (loose coupling): workers read/write a "
            "reducer-merged list; nobody calls anybody. Add a worker without touching "
            "the rest. *(This app's supervisor & parallel patterns.)*\n"
            "- **Handoffs via `Command`** (tight coupling): a node passes control "
            "*and* data in one return — `Command(goto=\"worker\", update={...})`.")

    with st.expander("7 · Context engineering — why long context degrades, and the fixes"):
        st.markdown(
            "Because the agent's memory *is* the transcript, every loop makes the "
            "context longer — and longer context is **slower, more expensive, and "
            "eventually overflows** the window and errors. Worse, models attend less "
            "reliably to the middle of a very long context ('lost in the middle'). So "
            "you actively **engineer** what stays in the window. Fixes, cheapest first:\n\n"
            "- **Trim** — keep the system prompt + the most recent N tokens. "
            "Deterministic, no LLM call, but *drops* old facts.\n"
            "- **Summarize / compact** — replace old turns with an LLM-written recap. "
            "One extra call, but *keeps the meaning* of what happened.\n"
            "- **Offload** — write state to memory/a file and pull it back by "
            "reference, keeping the live window small.")
        st.caption("▶ Watch it: **Pattern 3** lets you switch trim / summarize / none "
                   "and shows the **before→after token trace** each loop — the gap is "
                   "real money and latency.")

    with st.expander("8 · The agent harness — what turns an LLM into a dependable agent"):
        st.markdown(
            "An LLM alone is a text predictor. The **harness** is everything around it "
            "that makes it trustworthy — and it's most of the real engineering:\n\n"
            "- **Typed tools with sharp descriptions** — the model's API to the world.\n"
            "- **Guardrails** — `max_steps` / `recursion_limit`, loop detection, "
            "budgets. Bound the loop or bankrupt yourself.\n"
            "- **Verification** — *never trust LLM arithmetic*. Deterministic code "
            "(`validate_path`, the verifier node) computes the true cost and provides "
            "the ground-truth stop.\n"
            "- **Structured extraction** — agents return chat; pull clean data back "
            "out (or force a JSON schema with `with_structured_output`).\n"
            "- **Observability** — log every tool call so nothing is a black box. "
            "*(That log is exactly what powers this app's live exploration view.)*\n"
            "- **Mix code + LLMs** — LLM for judgment, plain code for guarantees. Not "
            "every node needs to be a model.")
        st.caption("The mantra: **LLM proposes, harness disposes.** Give the model "
                   "autonomy where judgment helps; keep hard guarantees in code.")

    with st.expander("9 · Evaluation & ground truth — how you know it actually worked"):
        st.markdown(
            "The hardest part of agents in production is knowing whether the output is "
            "*right*. Most real tasks have no cheap oracle, so teams build eval sets, "
            "LLM-as-judge graders, and trajectory checks. This app sidesteps that on "
            "purpose: **shortest-path has a checkable ground truth** — Dijkstra — that "
            "the *harness* computes but the *agent never sees*.\n\n"
            "That's what makes it the ideal teaching task: the agent **can't see the "
            "whole graph** (so it must genuinely reason and explore — real agentic "
            "behaviour), yet every run is **objectively scorable** (optimal or not). "
            "You always know whether the agent truly succeeded or just sounded "
            "confident — the difference between a demo and a system.")

    with st.expander("10 · When to build an agent (and when NOT to)"):
        st.markdown(
            "**Reach for an agent when:** the task is open-ended, the number of steps "
            "is unknown up front, the path depends on what you discover along the way, "
            "and a fixed script would be brittle. Pathfinding-through-discovery is a "
            "textbook fit.\n\n"
            "**Do NOT build an agent when:** the task decomposes into a fixed sequence "
            "(use a workflow), a single well-prompted call suffices, or you can't "
            "afford the variance in cost and latency. **Start with the simplest thing "
            "that works** — one call, then a workflow — and add agency only where the "
            "task genuinely resists being scripted. Much of the craft is knowing which "
            "rung of the ladder to stop on.")

    st.success("You now have the whole conceptual toolkit. Head to **🤖 Run a Pattern** "
               "and watch each idea execute — the live exploration view makes every one "
               "of these concepts something you can *see*, not just read.")


# =========================================================================== #
# TAB: Learn LangGraph                                                        #
# =========================================================================== #
with tab_learn:
    st.markdown("## LangGraph in five ideas\n"
        "LangGraph models an agent app as a **state machine** — a graph of steps "
        "over a shared, evolving **state**. That's the whole idea; the rest is detail.")
    st.markdown("**1 · State** — a `TypedDict`; reducers say how fields merge.")
    st.code('''from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]   # add_messages APPENDS (this = memory)
    # a plain field (no Annotated) OVERWRITES on each update''', language="python")
    st.markdown("**2 · Nodes** — functions `state -> partial update`.")
    st.code('''def agent(state: State) -> dict:
    return {"messages": [model.invoke(state["messages"])]}  # merged by the reducer''',
            language="python")
    st.markdown("**3 · Edges** — control flow. Conditional edges pick the next node.")
    st.code('''g.add_edge(START, "agent")                        # unconditional
g.add_conditional_edges("agent", route, {         # route() returns a key...
    "tools": "tools", END: END,                   # ...mapped to the next node
})''', language="python")
    st.markdown("**4 · Compile & run** — invoke to completion, or stream each step.")
    st.code('''graph = g.compile()
graph.invoke({"messages": [...]})                       # run to completion
for step in graph.stream({...}, stream_mode="updates"): # watch node-by-node
    print(step)                                         # {node_name: partial_update}''',
            language="python")
    st.markdown("**5 · The advanced levers** (each used by a pattern in this app):")
    st.code('''from langgraph.prebuilt import ToolNode          # runs the LLM's tool calls
from langgraph.types import Command, Send, interrupt
from langgraph.checkpoint.memory import MemorySaver   # persistence -> memory/resume

ToolNode(tools)                       # pattern 1
Command(goto="worker", update={...})  # agent-to-agent handoff (control + data)
Send("branch", {"via": v})            # pattern 4: dynamic PARALLEL fan-out
interrupt({"path": p})                # pattern 5: pause for a human
g.compile(checkpointer=MemorySaver()) # required for interrupt/resume & memory''',
            language="python")
    st.info("Open `patterns.py` next to this tab — each of the 5 patterns is a "
            "short, commented `StateGraph`. The **Run a Pattern** tab has notes + "
            "copyable code for every one.")


# =========================================================================== #
# TAB: The Graph                                                              #
# =========================================================================== #
with tab_graph:
    st.subheader("The world the agents search")
    st.write(f"A random weighted **directed** graph. The agents must find the "
             f"cheapest path from **{env.source}** to **{env.target}**, but they can "
             f"only see it through the `get_neighbors` tool — one node at a time.")
    opt_cost, opt_path = env.dijkstra()
    show_answer = st.toggle("Reveal the answer key (Dijkstra)", value=True)
    st.graphviz_chart(problem_dot(env, highlight=opt_path if show_answer else None),
                      use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Adjacency (ground truth)**")
        st.code(env.to_text())
    with c2:
        st.markdown("**Optimum (harness only — agents don't see this)**")
        if show_answer and opt_path:
            st.success(f"{' → '.join(opt_path)}  ·  cost **{opt_cost}**")
        else:
            st.info("Answer hidden. The agent has to earn it.")


# =========================================================================== #
# Shared render helpers                                                        #
# =========================================================================== #
def inject_highlight(dot: str, active_names, visited_names) -> str:
    """Return the pattern's DOT with the active node(s) painted amber and the
    already-run node(s) greyed out — appended as override node statements so they
    win over the base styles. Stream node names map 1:1 to DOT ids, except the
    parallel pattern's 'branch' which fans out to b0, b1, ... boxes."""
    def ids_for(name: str) -> set[str]:
        if name == "branch":
            return set(re.findall(r"\b(b\d+)\b", dot))
        return {name} if re.search(r'(^|[\s"])' + re.escape(name) + r"\b", dot) else set()

    act, vis = set(), set()
    for n in active_names:
        act |= ids_for(n)
    for n in visited_names:
        vis |= ids_for(n)
    vis -= act
    lines = [f'  {i} [style="rounded,filled", fillcolor="#ededed", '
             f'color="gray70", fontcolor="gray45"];' for i in sorted(vis)]
    lines += [f'  {i} [style="rounded,filled", fillcolor="#ffd166", '
              f'color="#e07a00", penwidth=3];' for i in sorted(act)]
    return dot.rstrip()[:-1] + "\n" + "\n".join(lines) + "\n}"


# =========================================================================== #
# LIVE EXPLORATION — the white-box view of the agent's progress on the graph.   #
#                                                                              #
# The whole thesis of this app is that the agent can't see the graph up front; #
# it must DISCOVER it through the `get_neighbors` tool. We keep the COMPLETE    #
# graph on screen and colour-code it from the tool-call log: nodes light up as #
# the agent visits them, edges brighten as it inspects them, and its best path #
# is drawn bold. This is the gemniCode-style live view — watch the agent work. #
# =========================================================================== #
def parse_neighbor_result(result: str) -> list[tuple[str, int]]:
    """Turn a get_neighbors result string ('B:3, C:7') into [('B',3),('C',7)]."""
    out: list[tuple[str, int]] = []
    if not result or "no outgoing" in result:
        return out
    for part in result.split(","):
        v, _, w = part.strip().partition(":")
        v = v.strip()
        if not v:
            continue
        try:
            out.append((v, int(float(w.strip()))))
        except ValueError:
            out.append((v, 0))
    return out


def new_exploration(env) -> dict:
    """A fresh 'what the agent has seen so far' accumulator.

    source & target are *given* by the task statement, so they count as known
    from step zero — everything else must be earned with a tool call.
    """
    return {
        "discovered": {env.source, env.target},   # nodes the agent knows exist
        "queried": set(),                          # nodes it called get_neighbors on
        "edges": {},                               # (u,v) -> w  revealed edges only
        "current": env.source,                     # the agent's "position" (last touch)
        "candidate": None,                         # last path handed to submit_path
        "best_path": None,                         # cheapest VALID path so far
        "best_cost": None,
        "n_calls": 0,                              # tool calls processed
    }


def apply_log_entry(ex: dict, entry: dict) -> dict:
    """Fold one tool-call log record into the exploration accumulator."""
    tool = entry.get("tool")
    if tool == "think":          # reasoning, not a tool call — no fog/metric change
        return ex
    ex["n_calls"] += 1
    args = entry.get("args", {}) or {}
    result = str(entry.get("result", ""))
    if tool == "get_neighbors":
        node = args.get("node", "")
        if node:
            ex["queried"].add(node)
            ex["discovered"].add(node)
            ex["current"] = node
            for v, w in parse_neighbor_result(result):
                ex["edges"][(node, v)] = w
                ex["discovered"].add(v)
    elif tool == "submit_path":
        path = args.get("path") or []
        if isinstance(path, str):
            path = [p.strip() for p in path.split(",") if p.strip()]
        ex["candidate"] = path
        ex["discovered"].update(path)
        if path:
            ex["current"] = path[-1]
        ok = "valid=True" in result
        cost = None
        if "cost=" in result:
            try:
                cost = float(result.split("cost=")[1].split()[0])
            except (IndexError, ValueError):
                cost = None
        if ok and cost is not None and (ex["best_cost"] is None or cost < ex["best_cost"]):
            ex["best_cost"], ex["best_path"] = cost, path
    return ex


def exploration_from_log(env, log: list) -> dict:
    """Replay a whole log into one exploration snapshot (used for static summaries)."""
    ex = new_exploration(env)
    for entry in log or []:
        apply_log_entry(ex, entry)
    return ex


def explore_dot(env, ex: dict, static: bool = False) -> str:
    """Render the COMPLETE graph (every edge + weight always visible), colour-coded
    by the agent's progress: nodes light up as it discovers/visits them, edges
    brighten as it inspects them, and the best/candidate path is drawn bold.

    `static=True` drops the amber 'current node' glow — for the final still frame.
    """
    discovered, edges = ex["discovered"], ex["edges"]
    current = None if static else ex.get("current")
    best = set(zip(ex["best_path"], ex["best_path"][1:])) if ex.get("best_path") else set()
    cand = set(zip(ex["candidate"], ex["candidate"][1:])) if ex.get("candidate") else set()

    # Saturated fills + explicit fontcolors so it reads on BOTH light and dark themes.
    styles = {
        "current": 'fillcolor="#ffd166", color="#e0a100", penwidth=3.6, fontcolor="#3a2c00"',
        "source":  'fillcolor="#2ea043", color="#56d364", penwidth=2.6, fontcolor="white"',
        "target":  'fillcolor="#f85149", color="#ff7b72", penwidth=2.6, fontcolor="white"',
        "seen":    'fillcolor="#388bfd", color="#79c0ff", penwidth=2.2, fontcolor="white"',
        "unseen":  'fillcolor="#30363d", color="#8b949e", penwidth=1.6, fontcolor="#c9d1d9"',
    }
    lines = ['digraph { rankdir=LR; bgcolor="transparent"; nodesep=0.45; ranksep=0.8;',
             '  node [shape=circle, fontname=Helvetica, style="filled", penwidth=1.8];']
    for u in env.nodes():
        if u == current:
            role = "current"
        elif u == env.source:
            role = "source"
        elif u == env.target:
            role = "target"
        elif u in discovered:
            role = "seen"          # agent has visited / inspected this node
        else:
            role = "unseen"        # part of the graph, not yet reached (still visible)
        lines.append(f'  {u} [{styles[role]}];')
    # The WHOLE graph, with every weight, is always on screen. Colour by progress:
    #   best path (green) > current candidate (blue) > inspected via get_neighbors
    #   (bright) > not yet explored (dim, but fully visible).
    for u in env.nodes():
        for v, w in env.neighbors(u):
            if (u, v) in best:
                attr = 'color="#3fb950", penwidth=3.6, fontcolor="#56d364"'
            elif (u, v) in cand:
                attr = 'color="#58a6ff", penwidth=3.2, fontcolor="#79c0ff"'
            elif (u, v) in edges:                # inspected by the agent -> bright
                attr = 'color="#d0d7de", penwidth=2.2, fontcolor="#e6edf3"'
            else:                                 # not yet explored -> dim but visible
                attr = 'color="#484f58", penwidth=1.2, fontcolor="#768390"'
            lines.append(f'  {u} -> {v} [label="{int(w)}", {attr}];')
    lines.append("}")
    return "\n".join(lines)


GRAPH_LEGEND = (
    "🟢 source &nbsp; 🔴 target &nbsp; 🟡 agent's position &nbsp; "
    "🔵 visited &nbsp; ⚫ not yet reached &nbsp;·&nbsp; "
    "edges: dim = not explored, bright = inspected via `get_neighbors`, "
    "🟩 green = best valid path, 🟦 blue = last submission"
)


def ledger_line(entry: dict) -> str:
    """Format one log record as a white-box ledger line (a reason or observe step)."""
    tool = entry.get("tool")
    args = entry.get("args", {}) or {}
    result = str(entry.get("result", ""))
    if tool == "think":          # the model's own reasoning (chain of thought)
        return f"&nbsp;&nbsp;💭 *{result[:700]}*"
    if tool == "get_neighbors":
        return f"&nbsp;&nbsp;🔍 `get_neighbors({args.get('node', '?')})` → {result}"
    if tool == "submit_path":
        path = args.get("path") or []
        path = ",".join(path) if isinstance(path, list) else str(path)
        icon = "✅" if "valid=True" in result else "❌"
        return f"&nbsp;&nbsp;📤 `submit_path({path})` → {icon} {result}"
    return f"&nbsp;&nbsp;`{tool}` → {result}"


def node_narration(node: str, upd) -> list[str]:
    """White-box narration for a node's state update (the REASON step + decisions).

    Tool *results* come from the log (ledger_line); this surfaces everything else:
    the model's reasoning text, its intent to call a tool, and pattern-specific
    state (routing decisions, critiques, token traces, parallel branches)."""
    out: list[str] = []
    if not isinstance(upd, dict):
        return out
    for m in upd.get("messages", []) or []:
        # only narrate the model's OWN words (ai) — tool results come from the ledger
        if getattr(m, "type", "") == "ai":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                out.append(f"&nbsp;&nbsp;💭 *{txt[:700]}*")
            for c in getattr(m, "tool_calls", None) or []:
                out.append(f"&nbsp;&nbsp;🛠️ wants `{c['name']}({c['args']})`")
    labels = {"next": "routes to", "iteration": "iteration", "approved": "approved",
              "critique": "critique", "best_cost": "best cost", "proposed_path": "proposed"}
    for k, label in labels.items():
        if k in upd and upd[k] not in (None, "", [], False):
            out.append(f"&nbsp;&nbsp;· {label}: `{upd[k]}`")
    for r in upd.get("route_log", []) or []:
        out.append(f"&nbsp;&nbsp;· 🧭 supervisor suggested `{r.get('suggested')}`, "
                   f"harness applied `{r.get('applied')}`")
    for t in upd.get("token_trace", []) or []:
        out.append(f"&nbsp;&nbsp;· ✂️ context [{t.get('strategy')}]: "
                   f"{t.get('before')}→{t.get('after')} est. tokens")
    for c in upd.get("candidates", []) or []:
        p = " → ".join(c["path"]) if c.get("path") else "(failed)"
        out.append(f"&nbsp;&nbsp;· 🌿 branch via {c.get('via', '?')}: {p} (cost {c.get('cost')})")
    return out


def render_result(pattern_key: str, final_state: dict, log: list, env) -> None:
    opt_cost, opt_path = env.dijkstra()
    path = final_state.get("best_path")
    cost = final_state.get("best_cost")
    if path is None and "messages" in final_state:
        from agent_toolkit import extract_best_submission
        path, cost = extract_best_submission(final_state["messages"])
    if path is None:
        path, cost = final_state.get("path"), final_state.get("cost")

    st.markdown("### Result")
    if path:
        if cost is not None and opt_cost is not None and cost <= opt_cost:
            st.success(f"Agent found: **{' → '.join(path)}**  ·  cost **{cost}**  ·  ✅ optimal")
        else:
            st.warning(f"Agent found: **{' → '.join(path)}**  ·  cost **{cost}**  ·  "
                       f"⚠️ optimum is **{opt_cost}** ({' → '.join(opt_path)})")
        st.graphviz_chart(problem_dot(env, highlight=path), use_container_width=True)
    else:
        st.error("The agent never submitted a valid path. Try a stronger model, a "
                 "smaller graph, or temperature 0.")

    if pattern_key == "supervisor" and final_state.get("route_log"):
        st.markdown("### Supervisor routing — LLM proposes, harness disposes")
        st.dataframe(final_state["route_log"], use_container_width=True)
        st.caption("Where 'suggested' ≠ 'applied', the harness overrode the LLM to "
                   "guarantee progress and termination.")
    if pattern_key == "evaluator_optimizer":
        st.markdown(f"### Reflection loop — {final_state.get('iteration', 0)} iteration(s)")
        tt = final_state.get("token_trace", [])
        if tt:
            st.markdown("**Context-window management** — est. tokens carried forward, "
                        "before vs after each loop:")
            st.dataframe(tt, use_container_width=True)
            st.caption("Strategy 'none' grows every loop (context bloat); 'trim' and "
                       "'summarize' keep it bounded. That gap is real money + latency.")
    if pattern_key == "parallel_explorers" and final_state.get("candidates"):
        st.markdown("### Parallel branches (one explorer per first hop)")
        st.dataframe([{"first hop": c.get("via"),
                       "path": " → ".join(c["path"]) if c.get("path") else "(failed)",
                       "cost": c.get("cost")} for c in final_state["candidates"]],
                     use_container_width=True)

    if log:
        ex = exploration_from_log(env, log)
        st.markdown("### 🔦 The agent's exploration (complete graph, colour-coded)")
        st.caption("The whole graph, with the agent's trace painted on: **bright** "
                   "edges are the ones it inspected via `get_neighbors`, **dim** edges "
                   "it never bothered to look at, and the **green** path is its best "
                   "valid route. " + GRAPH_LEGEND)
        st.graphviz_chart(explore_dot(env, ex, static=True), use_container_width=True)
        seen = len(ex["discovered"])
        st.caption(f"Visited **{seen}/{len(env.nodes())}** nodes and inspected "
                   f"**{len(ex['edges'])}** edges across **{ex['n_calls']}** tool calls "
                   f"— often *without* looking at the whole graph.")

    st.markdown("### Tool-call log (observability)")
    tool_calls = [e for e in log if e.get("tool") != "think"]   # drop reasoning rows
    if tool_calls:
        st.dataframe(tool_calls, use_container_width=True,
                     height=min(400, 40 + 30 * len(tool_calls)))
        st.caption(f"{len(tool_calls)} tool calls — the agent's entire interaction "
                   f"with the world.")
    else:
        st.info("No tool calls recorded.")


def hitl_step(graph, inp, config):
    """Run a human-in-the-loop graph until it pauses at interrupt(); return the
    interrupt payload, or None if it finished."""
    payload = None
    for chunk in graph.stream(inp, config):
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
    return payload


# =========================================================================== #
# TAB: Run a Pattern                                                          #
# =========================================================================== #
with tab_run:
    if not PATTERNS:
        st.stop()
    choice = st.radio("Orchestration pattern", list(PATTERNS),
                      format_func=lambda k: PATTERNS[k].title)
    p = PATTERNS[choice]
    st.markdown(f"#### {p.title}\n*{p.tagline}*")
    st.markdown(p.theory)
    st.markdown("**Concepts:** " + "  ".join(f"`{c}`" for c in p.concepts))

    st.markdown("**How this graph is wired:**")
    st.graphviz_chart(p.dot(env), use_container_width=True)

    with st.expander("📓 Notes — read / copy into your notebook", expanded=True):
        st.markdown(p.notes)
    with st.expander("📋 Code — build this graph (copy me)", expanded=False):
        st.code(p.snippet, language="python")

    st.divider()

    # ---- Interactive human-in-the-loop flow ---------------------------------
    if p.interactive:
        st.info("**Human-in-the-loop.** The agent proposes a path, then PAUSES via "
                "`interrupt()` for you to approve or reject. State is persisted by a "
                "checkpointer so the run resumes exactly where it paused.")
        if st.button("▶️ Start / restart run", type="primary"):
            log: list[dict] = []
            try:
                graph = p.build(make_llm(), env, log, {})
                cfg = {"configurable": {"thread_id": f"hitl-{time.time()}"},
                       "recursion_limit": 50}
                with st.spinner("Agent is proposing a path…"):
                    pending = hitl_step(graph, p.init(env), cfg)
                st.session_state.hitl = {"graph": graph, "cfg": cfg, "log": log,
                                         "pending": pending, "done": pending is None}
            except Exception as e:  # noqa: BLE001
                st.error(f"Run failed: `{type(e).__name__}: {e}` — is Ollama running "
                         f"and `{model}` pulled?")
            st.rerun()

        h = st.session_state.get("hitl")
        if h and h.get("pending"):
            proposed = h["pending"].get("proposed_path")
            st.markdown("### The agentic system (🟡 = waiting on you at `review`)")
            st.graphviz_chart(
                inject_highlight(p.dot(env), {"review"}, {"propose"}),
                use_container_width=True)
            st.markdown(f"### 🧑‍⚖️ Your review (attempt {h['pending'].get('attempt')})")
            if proposed:
                ok, cost, _ = env.validate_path(proposed)
                st.write(f"The agent proposes **{' → '.join(proposed)}** "
                         f"(valid={ok}, cost={cost}).")
                st.graphviz_chart(problem_dot(env, highlight=proposed), use_container_width=True)
            else:
                st.warning("The agent didn't produce a valid path this attempt.")
            with st.expander("🔦 The agent's exploration so far (complete graph, colour-coded)"):
                st.caption(GRAPH_LEGEND)
                st.graphviz_chart(
                    explore_dot(env, exploration_from_log(env, h["log"]), static=True),
                    use_container_width=True)
            fb = st.text_input("Rejection feedback (optional, sent back to the agent)",
                               key="hitl_fb")
            col_a, col_r = st.columns(2)
            if col_a.button("✅ Approve", type="primary"):
                with st.spinner("Resuming after approval…"):
                    pending = hitl_step(h["graph"], Command(resume={"approved": True}), h["cfg"])
                h["pending"], h["done"] = pending, pending is None
                st.rerun()
            if col_r.button("❌ Reject & retry"):
                with st.spinner("Resuming with your feedback…"):
                    pending = hitl_step(h["graph"],
                                        Command(resume={"approved": False, "feedback": fb}),
                                        h["cfg"])
                h["pending"], h["done"] = pending, pending is None
                st.rerun()
        elif h and h.get("done"):
            st.markdown("### The agentic system (run complete)")
            st.graphviz_chart(
                inject_highlight(p.dot(env), {"finalize"}, {"propose", "review"}),
                use_container_width=True)
            final = h["graph"].get_state(h["cfg"]).values  # checkpointer -> final state
            render_result(choice, final, h["log"], env)

    # ---- Standard streaming run ---------------------------------------------
    else:
        context_strategy = None
        if choice == "evaluator_optimizer":
            context_strategy = st.selectbox(
                "Context-window strategy (watch the token trace change)",
                ["trim", "summarize", "none"],
                help="trim = drop old messages (no LLM call); summarize = LLM recap; "
                     "none = let it grow (baseline).")
        speed = st.slider("🎞️ Playback delay per step (seconds) — slow it down to *watch*",
                          0.0, 1.5, 0.35, 0.05,
                          help="Replays each reasoning step and tool call one at a time "
                               "so you can watch the graph light up. 0 = instant.")
        if st.button("▶️ Run this pattern", type="primary"):
            log = []
            try:
                opts = {"context_strategy": context_strategy} if context_strategy else {}
                graph = p.build(make_llm(), env, log, opts)

                # ---- white-box live layout: exploration | wiring, metrics, ledger ----
                st.markdown("### 🔦 Live white-box run")
                st.caption("**Left:** the complete graph, colour-coded as the agent "
                           "explores — nodes light up as it visits them, edges brighten "
                           "as it inspects them via `get_neighbors`, and its best path "
                           "is drawn bold. **Right:** which orchestration node is firing.")
                col_ex, col_wire = st.columns([3, 2], gap="large")
                with col_ex:
                    st.markdown("**Live graph exploration**")
                    explore_ph = st.empty()
                    st.caption(GRAPH_LEGEND)
                with col_wire:
                    st.markdown("**Orchestration wiring** (🟡 active · ⬜ done)")
                    wiring_ph = st.empty()
                mc = st.columns(4)
                m_calls, m_seen, m_best, m_opt = (c.empty() for c in mc)
                st.markdown("**📜 White-box ledger — how the model is actually finding "
                            "the path.** &nbsp; 💭 = the model's own reasoning · "
                            "🛠️ = a tool it decides to call · 🔍 `get_neighbors` / "
                            "📤 `submit_path` = the tool result it observes. Read it "
                            "top-to-bottom to follow reason → act → observe.")
                ledger_ph = st.empty()

                # initial frames + ground truth (harness-only) for the scoreboard
                ex = new_exploration(env)
                opt_cost, _ = env.dijkstra()
                explore_ph.graphviz_chart(explore_dot(env, ex), use_container_width=True)
                wiring_ph.graphviz_chart(p.dot(env), use_container_width=True)

                def paint_metrics():
                    m_calls.metric("🔧 Tool calls", ex["n_calls"])
                    m_seen.metric("🗺️ Nodes discovered", f"{len(ex['discovered'])}/{len(env.nodes())}")
                    bc = ex["best_cost"]
                    m_best.metric("💰 Best valid cost", "—" if bc is None else int(bc),
                                  delta=None if bc is None else int(bc - opt_cost),
                                  delta_color="inverse")
                    m_opt.metric("🎯 True optimum", "—" if opt_cost is None else int(opt_cost))
                paint_metrics()

                visited: set[str] = set()
                ledger: list[str] = []
                seen_logs = 0
                final_state: dict = {}

                def flush_ledger():
                    ledger_ph.markdown("\n\n".join(ledger[-60:]))

                with st.spinner("Agent is calling the model + tools…"):
                    for mode, chunk in graph.stream(
                        p.init(env), {"recursion_limit": 100},
                        stream_mode=["updates", "values"],
                    ):
                        if mode == "values":
                            final_state = chunk
                            continue
                        active = set(chunk.keys())
                        wiring_ph.graphviz_chart(
                            inject_highlight(p.dot(env), active, visited),
                            use_container_width=True)
                        visited |= active
                        for node, upd in chunk.items():
                            ledger.append(f"**▶ `{node}`**")
                            ledger += node_narration(node, upd)
                            flush_ledger()
                            if speed:
                                time.sleep(min(speed, 0.4))
                            # replay THIS node's tool calls into the exploration view
                            while seen_logs < len(log):
                                apply_log_entry(ex, log[seen_logs])
                                ledger.append(ledger_line(log[seen_logs]))
                                seen_logs += 1
                                explore_ph.graphviz_chart(
                                    explore_dot(env, ex), use_container_width=True)
                                paint_metrics()
                                flush_ledger()
                                if speed:
                                    time.sleep(speed)

                # drain any trailing tool calls, then settle both diagrams
                while seen_logs < len(log):
                    apply_log_entry(ex, log[seen_logs]); seen_logs += 1
                paint_metrics()
                explore_ph.graphviz_chart(explore_dot(env, ex, static=True),
                                          use_container_width=True)
                wiring_ph.graphviz_chart(
                    inject_highlight(p.dot(env), set(), visited), use_container_width=True)
                render_result(choice, final_state, log, env)
            except Exception as e:  # noqa: BLE001
                st.error(f"Run failed: `{type(e).__name__}: {e}`\n\nCommon causes: "
                         f"Ollama not running (`ollama serve`), model not pulled "
                         f"(`ollama pull {model}`), or the model can't call tools. "
                         f"Fast tool-capable picks: qwen2.5:3b-instruct, llama3.2:3b "
                         f"(or qwen2.5:7b-instruct for stronger, slower reasoning).")


# =========================================================================== #
# TAB: Deep Dives (concepts + snippet library)                               #
# =========================================================================== #
with tab_deep:
    st.markdown("## The concepts, in depth — with copyable snippets")

    with st.expander("🧠 Agentic reasoning (ReAct)", expanded=True):
        st.markdown(
            "A plain LLM maps text→text. An **agent** interleaves *reasoning* and "
            "*acting*: think → call a tool → read the observation → think again. "
            "This is **ReAct**. The entire 'agent' is this ~10-line loop; every "
            "framework is sugar over it. It shines on **discovery** tasks — the "
            "agent doesn't start with all the facts (it can't see the whole graph).")
        st.code('''def run_react(model, tools, messages, max_steps=8):
    by_name = {t.name: t for t in tools}
    convo = list(messages)
    for _ in range(max_steps):                 # <- guardrail: always cap the loop
        ai = model.invoke(convo); convo.append(ai)
        if not ai.tool_calls:                  # no tool -> the agent is done
            break
        for call in ai.tool_calls:             # act
            result = by_name[call["name"]].invoke(call["args"])   # observe
            convo.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return convo''', language="python")

    with st.expander("🤝 Agent-to-agent (A2A) interaction"):
        st.markdown(
            "Two ways agents talk in LangGraph. **Shared-state channels** (loose "
            "coupling): workers read/write a reducer-merged list. **Handoffs via "
            "`Command`** (tight): a node passes control *and* data in one return.")
        st.code('''# (a) shared state: a worker appends; another reads later
class State(TypedDict):
    candidates: Annotated[list, operator.add]   # merge concurrent writes
def worker(state): return {"candidates": [my_proposal]}

# (b) explicit handoff: update state AND choose the next agent in one step
from langgraph.types import Command
def planner(state):
    return Command(goto="researcher", update={"task": "look up X"})''', language="python")

    with st.expander("✂️ Context-window shrinking"):
        st.markdown(
            "Every model has a fixed context window. A long run appends messages "
            "until it's slow, costly, and finally overflows. Fixes, cheapest first: "
            "**trim** (drop old, no LLM call) → **summarize** (LLM recap, keeps "
            "meaning) → **offload** (write to memory/file, reference later).")
        st.code('''from langchain_core.messages import trim_messages
# TRIM — keep system prompt + most recent tokens, deterministic, free
kept = trim_messages(history, token_counter=count, max_tokens=300,
                     strategy="last", include_system=True, start_on="human")

# SUMMARIZE — one LLM call, preserves meaning of the dropped turns
def summarize(llm, history, keep_last=2):
    head, tail = history[:-keep_last], history[-keep_last:]
    recap = llm.invoke([SystemMessage("Summarize progress in 3-4 terse bullets."),
                        HumanMessage("\\n".join(str(m.content) for m in head))])
    return [SystemMessage("Summary:\\n" + recap.content)] + tail''', language="python")

    with st.expander("💾 Memory, persistence & human-in-the-loop"):
        st.markdown(
            "A **checkpointer** saves graph state at every step. That single change "
            "unlocks: **resume** after a crash, **time-travel** to a past state, and "
            "**human-in-the-loop** via `interrupt()`. The run config's `thread_id` is "
            "the key the saved state is stored under.")
        st.code('''from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

def review(state):
    decision = interrupt({"path": state["proposed_path"]})   # PAUSE, surface to app
    return {"approved": decision["approved"]}

graph  = g.compile(checkpointer=MemorySaver())               # persistence
config = {"configurable": {"thread_id": "run-1"}}            # state key

for chunk in graph.stream(init, config):                     # runs until it pauses
    if "__interrupt__" in chunk:
        payload = chunk["__interrupt__"][0].value
graph.invoke(Command(resume={"approved": True}), config)     # resume where it paused''',
                language="python")

    with st.expander("🔧 Harness engineering (what makes an LLM dependable)"):
        st.markdown(
            "- **Typed tools** with good docstrings (the model reads them to choose).\n"
            "- **Guardrails**: `max_steps`, `recursion_limit`, loop detection.\n"
            "- **Verification**: never trust LLM arithmetic — `submit_path` and the "
            "deterministic `verifier` compute the true cost; ground truth stops loops.\n"
            "- **Structured extraction**: agents return chat; pull clean data back out.\n"
            "- **Observability**: the shared `log` records every tool call.\n"
            "- **Mix code + LLMs**: LLM for judgment, plain code for guarantees.")
        st.code('''from langchain_core.tools import tool

@tool
def submit_path(path: str) -> str:
    """Submit a route 'A,C,E'. Returns validity + exact cost. ALWAYS end with this
    — never state a cost you computed yourself."""       # docstring = the model's spec
    nodes = [p.strip() for p in path.split(",") if p.strip()]
    ok, cost, note = env.validate_path(nodes)            # DETERMINISTIC ground truth
    return f"valid={ok} cost={cost} note={note}"

# structured output (more robust than regex-scraping the transcript):
# structured = llm.with_structured_output(MyPydanticModel)''', language="python")

    with st.expander("🗺️ Streaming — watch the agent think"):
        st.markdown("`stream_mode` controls what you get per step. Use a list to get "
                    "both live node updates AND the full final state.")
        st.code('''for mode, chunk in graph.stream(init, config, stream_mode=["updates", "values"]):
    if mode == "updates":       # {node_name: partial_update} — great for a live UI
        show(chunk)
    elif mode == "values":      # the FULL state after that step; last one = final
        final_state = chunk
# other modes: "messages" (token-by-token LLM output), "debug" (everything)''',
                language="python")

    st.markdown("### Where to go next")
    st.markdown(
        "- Add a **summarization vs trim** A/B on a long graph and compare token traces.\n"
        "- Give the parallel pattern a **beam width** (keep top-k branches, prune the rest).\n"
        "- Swap `MemorySaver` for a **SqliteSaver** so runs survive a restart.\n"
        "- Replace the toy graph with a real one (road network, dependency DAG) — the "
        "agents don't change.")
