"""patterns.py — five LangGraph orchestration patterns, simple to advanced.

Each pattern solves the SAME task (find the cheapest path) but wires agents
together differently. Studying them side by side is the fastest way to learn
LangGraph and multi-agent design.

  1. single_react        — ONE agent in a ReAct loop.            (the atom)
  2. supervisor          — a router delegating to WORKER agents.  (orchestration)
  3. evaluator_optimizer — propose/critique REFLECTION loop       (self-correction
                            + context-window trimming / summarizing)
  4. parallel_explorers  — MAP-REDUCE fan-out with Send.          (parallelism)
  5. human_in_the_loop   — pause for human approval via interrupt (memory +
                            checkpointer + control)

LangGraph mental model (this is the whole API, really):
  * A `StateGraph` is a state machine over a shared STATE (a TypedDict).
  * NODES are functions `state -> partial_state_update`.
  * EDGES pick the next node; CONDITIONAL EDGES pick it from a function's return.
  * REDUCERS (`Annotated[list, add]`) merge concurrent/repeated field updates.
  * `.compile()` returns a runnable you can `.invoke()` or `.stream()`.
  * A CHECKPOINTER persists state between steps -> enables interrupt/resume, memory.

Every `Pattern` carries `notes` (for the UI's note-taking panel) and `snippet`
(a copy-paste code sketch), so the app can teach each one in place.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt

from agent_toolkit import (
    estimate_tokens,
    explorer_messages,
    extract_best_submission,
    make_tools,
    record_thoughts,
    run_react,
    summarize_history,
    trim_history,
)
from graph_env import GraphEnv


@dataclass
class Pattern:
    """Everything the UI needs to teach + run one orchestration pattern."""

    key: str
    title: str
    tagline: str
    concepts: list[str]
    theory: str
    notes: str                                    # detailed markdown for note-taking
    snippet: str                                  # copy-paste code sketch
    build: Callable[..., Any]                     # (llm, env, log, opts=None) -> graph
    init: Callable[[GraphEnv], dict]
    dot: Callable[[GraphEnv], str]
    interactive: bool = False                     # True for human_in_the_loop


# =========================================================================== #
# PATTERN 1 — SINGLE REACT AGENT                                              #
# =========================================================================== #
class ReactState(TypedDict):
    # add_messages APPENDS new messages instead of overwriting the list — that
    # one reducer is why the agent "remembers" the conversation.
    messages: Annotated[list, add_messages]
    steps: int                                    # agent turns taken (loop guardrail)


# GUARDRAIL: an exhaustive DFS prompt can make a (small) model loop without ever
# emitting a final answer. Cap the agent turns well below the recursion limit so we
# stop cleanly and keep the best path found so far, instead of crashing.
REACT_STEP_CAP = 24


def build_single_react(llm, env: GraphEnv, log: list, opts: dict | None = None):
    tools = make_tools(env, log)
    model = llm.bind_tools(tools)  # hand the model the tool schemas

    def agent(state: ReactState) -> dict:
        return {"messages": [model.invoke(state["messages"])],
                "steps": state.get("steps", 0) + 1}

    def route(state: ReactState) -> str:
        if state.get("steps", 0) >= REACT_STEP_CAP:
            return END                            # loop guard fired -> stop, keep best
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        # The agent wants to finish. GUARDRAIL: don't accept an answer it never
        # verified — if it hasn't submitted a valid path yet, make it submit first.
        best_path, _ = extract_best_submission(state["messages"])
        return END if best_path is not None else "nudge"

    def nudge(state: ReactState) -> dict:
        # Force tool use before finishing (small models like to answer in prose).
        return {"messages": [HumanMessage(content=
            f"You have not confirmed a path yet. Call submit_path now with your best "
            f"complete route from {env.source} to {env.target} to validate its true "
            f"cost before you finish.")]}

    g = StateGraph(ReactState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))      # prebuilt: executes tool calls
    g.add_node("nudge", nudge)                # guardrail: require a verified submission
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route,
                            {"tools": "tools", "nudge": "nudge", END: END})
    g.add_edge("tools", "agent")              # loop: tool result -> think again
    g.add_edge("nudge", "agent")              # remind, then let it act
    return g.compile()


def init_single_react(env: GraphEnv) -> dict:
    return {"messages": explorer_messages(env), "steps": 0}


def dot_single_react(env: GraphEnv) -> str:
    return """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
  START [shape=circle, label="", width=0.2, style=filled, fillcolor=black];
  END   [shape=doublecircle, label="", width=0.2];
  agent [label="agent\\n(LLM reasons)"];
  tools [label="tools\\n(get_neighbors,\\nsubmit_path)"];
  START -> agent;
  agent -> tools [label="tool_calls?", style=dashed];
  tools -> agent;
  agent -> END [label="no tool -> done", style=dashed];
}"""


NOTES_REACT = """
**What it is.** The atom of agentic AI: one node calls the LLM, a `ToolNode`
executes any tool the LLM asked for, and control loops back until the LLM answers
*without* a tool call.

**Why the loop matters.** A plain LLM does text→text. An agent interleaves
reasoning and acting: think → call `get_neighbors` → read the result → think
again. That's **ReAct** (Reason + Act). Discovery tasks (the agent can't see the
whole graph) are exactly where this shines.

**Key LangGraph pieces**
- `StateGraph(State)` — the state machine.
- `Annotated[list, add_messages]` — the reducer that appends messages (memory).
- `add_conditional_edges(node, router, mapping)` — router returns a key; the
  mapping says which node runs next. Here it decides "run a tool" vs "stop".
- `ToolNode(tools)` — prebuilt node that runs the `tool_calls` on the last message.

**Gotchas**
- Bind tools once (`llm.bind_tools(tools)`); the model must be a tool-calling model.
- Always cap the loop (recursion_limit / max_steps) or a confused model spins.
"""

SNIPPET_REACT = '''from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    messages: Annotated[list, add_messages]   # reducer = append (this is memory)

model = llm.bind_tools(tools)                 # tools = [get_neighbors, submit_path]

def agent(state):                             # a node: state -> partial update
    return {"messages": [model.invoke(state["messages"])]}

def route(state):                             # conditional edge router
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))          # executes the tool calls
g.add_edge(START, "agent")
g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
g.add_edge("tools", "agent")                  # the ReAct loop
graph = g.compile()
graph.invoke({"messages": [system, human]})
'''


# =========================================================================== #
# PATTERN 2 — SUPERVISOR / ORCHESTRATOR-WORKER (multi-agent)                  #
# =========================================================================== #
class SupervisorState(TypedDict):
    candidates: Annotated[list, operator.add]   # each worker appends its proposal
    explored: Annotated[list, operator.add]     # which workers have run
    route_log: Annotated[list, operator.add]    # {suggested, applied} for the UI
    best_path: Optional[list]
    best_cost: Optional[float]
    optimum_cost: Optional[float]
    next: str


def _explore_once(llm, env, log, style: str) -> dict:
    tools = make_tools(env, log)
    model = llm.bind_tools(tools)
    transcript = run_react(model, tools, explorer_messages(env, extra=style),
                           max_steps=14, on_step=record_thoughts(log))
    path, cost = extract_best_submission(transcript)
    return {"path": path, "cost": cost}


def build_supervisor(llm, env: GraphEnv, log: list, opts: dict | None = None):
    GREEDY = "Strategy: be GREEDY — at each node take the cheapest available edge."
    CAUTIOUS = ("Strategy: be CAUTIOUS — a cheap first edge can trap you. Compare "
                "at least two different first moves before committing.")
    LEGAL = {"GREEDY", "CAUTIOUS", "VERIFY", "FINISH"}

    def _default(state: SupervisorState) -> str:
        have = set(state.get("explored", []))
        if "greedy" not in have:
            return "GREEDY"
        if "cautious" not in have:
            return "CAUTIOUS"
        if state.get("best_path") is None:
            return "VERIFY"
        return "FINISH"

    def supervisor(state: SupervisorState) -> dict:
        default = _default(state)
        status = (f"candidates={len(state.get('candidates', []))}; "
                  f"workers_run={state.get('explored', [])}; "
                  f"verified={state.get('best_path') is not None}.")
        try:
            raw = str(llm.invoke([
                SystemMessage(content="You are a SUPERVISOR coordinating workers to "
                              "find a cheapest graph path. Workers: GREEDY, CAUTIOUS, "
                              "VERIFY, FINISH. Reply with EXACTLY one of those tokens."),
                HumanMessage(content=status + " Which worker acts next?"),
            ]).content).strip().upper()
            suggested = next((t for t in LEGAL if t in raw), "?")
        except Exception:
            suggested = "?"
        # LLM proposes, harness disposes: enforce a legal, terminating order.
        return {"next": default,
                "route_log": [{"suggested": suggested, "applied": default}]}

    def greedy(state: SupervisorState) -> dict:
        return {"candidates": [{"by": "greedy", **_explore_once(llm, env, log, GREEDY)}],
                "explored": ["greedy"]}

    def cautious(state: SupervisorState) -> dict:
        return {"candidates": [{"by": "cautious", **_explore_once(llm, env, log, CAUTIOUS)}],
                "explored": ["cautious"]}

    def verifier(state: SupervisorState) -> dict:
        # Deterministic worker: recompute TRUE costs, pick the best. Not every
        # agent must be an LLM — use code where you need guarantees.
        best_path, best_cost = None, None
        for c in state.get("candidates", []):
            if c.get("path") is None:
                continue
            ok, cost, _ = env.validate_path(c["path"])
            if ok and (best_cost is None or cost < best_cost):
                best_cost, best_path = cost, c["path"]
        opt_cost, _ = env.dijkstra()
        return {"best_path": best_path, "best_cost": best_cost, "optimum_cost": opt_cost}

    def route(state: SupervisorState) -> str:
        return {"GREEDY": "greedy", "CAUTIOUS": "cautious",
                "VERIFY": "verifier", "FINISH": END}[state["next"]]

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("greedy", greedy)
    g.add_node("cautious", cautious)
    g.add_node("verifier", verifier)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route,
        {"greedy": "greedy", "cautious": "cautious", "verifier": "verifier", END: END})
    for worker in ("greedy", "cautious", "verifier"):
        g.add_edge(worker, "supervisor")   # every worker reports back to the boss
    return g.compile()


def init_supervisor(env: GraphEnv) -> dict:
    return {"candidates": [], "explored": [], "route_log": [],
            "best_path": None, "best_cost": None, "optimum_cost": None, "next": ""}


def dot_supervisor(env: GraphEnv) -> str:
    return """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
  START [shape=circle, label="", width=0.2, style=filled, fillcolor=black];
  END   [shape=doublecircle, label="", width=0.2];
  supervisor [label="supervisor\\n(LLM routes)", style="rounded,filled", fillcolor="#ffe9c7"];
  greedy   [label="greedy\\nexplorer (LLM)"];
  cautious [label="cautious\\nexplorer (LLM)"];
  verifier [label="verifier\\n(pure code)", style="rounded,filled", fillcolor="#d7f0d7"];
  START -> supervisor;
  supervisor -> greedy;  supervisor -> cautious;  supervisor -> verifier;
  greedy -> supervisor;  cautious -> supervisor;  verifier -> supervisor;
  supervisor -> END [label="FINISH", style=dashed];
}"""


NOTES_SUPERVISOR = """
**What it is.** A **supervisor** node decides which specialist WORKER runs next,
loops through them, then finishes. Workers here: a greedy explorer (LLM), a
cautious explorer (LLM), and a verifier (**pure code**).

**Agent-to-agent (A2A) interaction.** The workers never call each other. They
communicate through **shared-state channels** — the `candidates` list, merged by
the `operator.add` reducer. Loose coupling: add a worker without touching the
others.

**LLM proposes, harness disposes.** The supervisor *asks the LLM* which worker to
run, but the harness enforces a legal, terminating order. This is the central
tension of agentic design: **autonomy vs. control**. Full LLM routing is flexible
but can loop/stall; hard-coded routing is reliable but rigid; this splits it.

**Key LangGraph pieces**
- Multiple `Annotated[..., operator.add]` channels for concurrent worker outputs.
- A routing node + `add_conditional_edges` back to workers (a hub-and-spoke shape).
- Mixing LLM nodes and plain-code nodes in one graph.

**Gotchas**
- Give the router a bounded, enumerable set of choices, and a default that always
  makes forward progress — otherwise multi-agent graphs can cycle forever.
"""

SNIPPET_SUPERVISOR = '''import operator
from typing import Annotated, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    candidates: Annotated[list, operator.add]   # workers APPEND here (A2A channel)
    explored:   Annotated[list, operator.add]
    best_path:  Optional[list]
    next: str

def supervisor(state):
    # ask the LLM, but fall back to a rule that guarantees progress
    default = ("greedy" if "greedy" not in state["explored"]
               else "verify" if state["best_path"] is None else "FINISH")
    return {"next": default}

def greedy(state):   return {"candidates": [explore(...)], "explored": ["greedy"]}
def verify(state):   return {"best_path": pick_cheapest(state["candidates"])}

def route(state):    return {"greedy": "greedy", "verify": "verify",
                             "FINISH": END}[state["next"]]

g = StateGraph(State)
for name, fn in [("supervisor", supervisor), ("greedy", greedy), ("verify", verify)]:
    g.add_node(name, fn)
g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", route,
                        {"greedy": "greedy", "verify": "verify", END: END})
g.add_edge("greedy", "supervisor")            # workers report back to the boss
g.add_edge("verify", "supervisor")
graph = g.compile()
'''


# =========================================================================== #
# PATTERN 3 — EVALUATOR / OPTIMIZER (reflection + context shrinking)          #
# =========================================================================== #
class EvalState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    path: Optional[list]
    cost: Optional[float]
    critique: str
    approved: bool
    optimum_cost: Optional[float]
    token_trace: Annotated[list, operator.add]  # [{iter, before, after, strategy}]


MAX_ITERS = 3


def build_evaluator_optimizer(llm, env: GraphEnv, log: list, opts: dict | None = None):
    opts = opts or {}
    strategy = opts.get("context_strategy", "trim")  # "trim" | "summarize" | "none"
    tools = make_tools(env, log)
    model = llm.bind_tools(tools)

    def optimizer(state: EvalState) -> dict:
        it = state.get("iteration", 0) + 1
        critique = state.get("critique", "")
        extra = "" if not critique else (
            f"A reviewer critiqued your last attempt (cost {state.get('cost')}): "
            f"{critique}\nProduce a cheaper valid path.")
        base = explorer_messages(env, extra=extra)

        # CONTEXT SHRINKING: bound the carried-forward history each loop.
        prior = state.get("messages", [])
        before = estimate_tokens(prior)
        if strategy == "summarize":
            kept = summarize_history(llm, prior, keep_last=2)
        elif strategy == "trim":
            kept = trim_history(prior, max_tokens=300)
        else:
            kept = prior
        after = estimate_tokens(kept)

        transcript = run_react(model, tools, base + kept, max_steps=14)
        path, cost = extract_best_submission(transcript)
        return {
            "messages": transcript[len(base):],  # only the new turns
            "iteration": it, "path": path, "cost": cost,
            "token_trace": [{"iter": it, "before": before, "after": after,
                             "strategy": strategy}],
        }

    def evaluator(state: EvalState) -> dict:
        opt_cost, _ = env.dijkstra()
        cost = state.get("cost")
        approved = cost is not None and opt_cost is not None and cost <= opt_cost
        critique = "Optimal — approved." if approved else ""
        if not approved and state.get("path"):
            per_edge = ", ".join(
                f"{u}->{v}:{int(env.edge_cost(u, v) or 0)}"
                for u, v in zip(state["path"], state["path"][1:]))
            crit = llm.invoke([
                SystemMessage(content="You review graph paths. In ONE sentence, name "
                              "the single edge that looks most wasteful and suggest "
                              "reconsidering the choice made just before it."),
                HumanMessage(content=f"Path edges/costs: {per_edge}. Total {cost}."),
            ])
            critique = str(crit.content).strip()
        return {"approved": approved, "critique": critique, "optimum_cost": opt_cost}

    def route(state: EvalState) -> str:
        if state.get("approved") or state.get("iteration", 0) >= MAX_ITERS:
            return END
        return "optimizer"

    g = StateGraph(EvalState)
    g.add_node("optimizer", optimizer)
    g.add_node("evaluator", evaluator)
    g.add_edge(START, "optimizer")
    g.add_edge("optimizer", "evaluator")
    g.add_conditional_edges("evaluator", route, {"optimizer": "optimizer", END: END})
    return g.compile()


def init_evaluator_optimizer(env: GraphEnv) -> dict:
    return {"messages": [], "iteration": 0, "path": None, "cost": None,
            "critique": "", "approved": False, "optimum_cost": None, "token_trace": []}


def dot_evaluator_optimizer(env: GraphEnv) -> str:
    return """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
  START [shape=circle, label="", width=0.2, style=filled, fillcolor=black];
  END   [shape=doublecircle, label="", width=0.2];
  optimizer [label="optimizer\\n(propose / improve)"];
  evaluator [label="evaluator\\n(critique + stop?)", style="rounded,filled", fillcolor="#ffe9c7"];
  START -> optimizer -> evaluator;
  evaluator -> optimizer [label="not optimal &\\niters left", style=dashed];
  evaluator -> END [label="approved or\\nmax iters", style=dashed];
}"""


NOTES_EVAL = """
**What it is.** A **reflection loop**: the optimizer proposes a path, the
evaluator critiques it and decides whether to stop or loop. Self-correction is a
core agentic capability.

**Two stop signals, on purpose**
- **Ground truth** (harness): is the cost optimal? -> approve. Deterministic.
- **Loop guard**: hit MAX_ITERS -> stop anyway. Never trust a model to always
  converge.

**Context-window shrinking (the headline).** Every loop appends messages. Left
alone, the history grows until it (a) slows down, (b) costs more, (c) overflows
the context window and errors. Three fixes, cheapest first:
- **trim** — keep the system prompt + most recent tokens (`trim_history`, no LLM call).
- **summarize** — replace old turns with an LLM recap (`summarize_history`, 1 call,
  keeps meaning).
- **none** — baseline, watch it grow.
Switch the strategy in the run panel and compare the **before→after token trace**.

**Key LangGraph pieces**
- A **cyclic** graph (`evaluator -> optimizer`) — loops are first-class.
- Conditional edge as the loop's exit test.
"""

SNIPPET_EVAL = '''from langchain_core.messages import trim_messages
# --- trimming: cheap, deterministic, no LLM call ---
kept = trim_messages(history, token_counter=count, max_tokens=300,
                     strategy="last", include_system=True, start_on="human")

# --- summarizing: one LLM call, preserves meaning of dropped turns ---
def summarize(llm, history, keep_last=2):
    head, tail = history[:-keep_last], history[-keep_last:]
    recap = llm.invoke([SystemMessage("Summarize progress in 3-4 terse bullets."),
                        HumanMessage("\\n".join(str(m.content) for m in head))])
    return [SystemMessage("Summary:\\n"+recap.content)] + tail

# --- the loop: propose -> evaluate -> (loop | stop) ---
def route(state):
    return END if state["approved"] or state["iteration"] >= 3 else "optimizer"
g.add_conditional_edges("evaluator", route, {"optimizer": "optimizer", END: END})
'''


# =========================================================================== #
# PATTERN 4 — PARALLEL EXPLORERS (map-reduce with Send)                       #
# =========================================================================== #
class ParState(TypedDict):
    via: str                                    # per-branch input (the first hop)
    candidates: Annotated[list, operator.add]   # reducer merges parallel results
    best_path: Optional[list]
    best_cost: Optional[float]
    optimum_cost: Optional[float]


def build_parallel_explorers(llm, env: GraphEnv, log: list, opts: dict | None = None):
    def dispatch(state: ParState) -> dict:
        return {}

    def fan_out(state: ParState) -> list[Send]:
        firsts = [v for v, _ in env.neighbors(env.source)]
        return [Send("branch", {"via": v}) for v in firsts] or [Send("reduce", {})]

    def branch(state: ParState) -> dict:
        via = state["via"]
        tools = make_tools(env, log)
        model = llm.bind_tools(tools)
        extra = (f"Your path MUST begin {env.source} -> {via}. From {via}, find the "
                 f"cheapest completion to {env.target}.")
        transcript = run_react(model, tools, explorer_messages(env, extra=extra),
                               max_steps=14, on_step=record_thoughts(log))
        path, cost = extract_best_submission(transcript)
        return {"candidates": [{"via": via, "path": path, "cost": cost}]}

    def reduce(state: ParState) -> dict:
        best_path, best_cost = None, None
        for c in state.get("candidates", []):
            if c.get("path") is None:
                continue
            ok, cost, _ = env.validate_path(c["path"])
            if ok and (best_cost is None or cost < best_cost):
                best_cost, best_path = cost, c["path"]
        opt_cost, _ = env.dijkstra()
        return {"best_path": best_path, "best_cost": best_cost, "optimum_cost": opt_cost}

    g = StateGraph(ParState)
    g.add_node("dispatch", dispatch)
    g.add_node("branch", branch)
    g.add_node("reduce", reduce)
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", fan_out, ["branch", "reduce"])
    g.add_edge("branch", "reduce")   # every branch fans IN to reduce (a barrier)
    g.add_edge("reduce", END)
    return g.compile()


def init_parallel_explorers(env: GraphEnv) -> dict:
    return {"via": "", "candidates": [], "best_path": None,
            "best_cost": None, "optimum_cost": None}


def dot_parallel_explorers(env: GraphEnv) -> str:
    firsts = [v for v, _ in env.neighbors(env.source)] or ["?"]
    branches = "\n".join(
        f'  b{i} [label="branch\\nvia {v}"]; dispatch -> b{i} [style=dashed]; b{i} -> reduce;'
        for i, v in enumerate(firsts))
    return f"""digraph {{
  rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
  START [shape=circle, label="", width=0.2, style=filled, fillcolor=black];
  END   [shape=doublecircle, label="", width=0.2];
  dispatch [label="dispatch\\n(fan-out: one\\nSend per 1st hop)", style="rounded,filled", fillcolor="#ffe9c7"];
  reduce   [label="reduce\\n(pick cheapest)", style="rounded,filled", fillcolor="#d7f0d7"];
  START -> dispatch;
{branches}
  reduce -> END;
}}"""


NOTES_PARALLEL = """
**What it is.** Map-reduce for agents. `Send` schedules many **parallel** copies
of a node, each with its own slice of work — here, one explorer per first-hop
neighbor of the source. They run concurrently, then **fan in** to `reduce`, which
merges their results and picks the cheapest.

**Key LangGraph pieces**
- **`Send(node, payload)`** returned from a conditional edge = dynamic fan-out. The
  number of branches is decided at runtime from state.
- **Reducer aggregation**: all branches write to the same `candidates` channel;
  `operator.add` merges the concurrent writes without a race.
- **Fan-in barrier**: because every `branch` edges to `reduce`, LangGraph waits for
  all branches before running `reduce`.

**When to use.** Independent sub-problems you want to explore at once (beam search,
multi-tool queries, ensembling). Watch out for cost: N branches = N× the tokens.

**Extension.** Add a beam width — keep only the top-k branches and prune the rest.
That's combinatorial search meeting agentic control.
"""

SNIPPET_PARALLEL = '''from langgraph.types import Send

def fan_out(state):                      # returned from a conditional edge
    return [Send("branch", {"via": v})   # one parallel branch per first hop
            for v in first_hops]

def branch(state):                       # runs in parallel; own state slice
    result = explore_from(state["via"])
    return {"candidates": [result]}      # reducer merges concurrent writes

g.add_conditional_edges("dispatch", fan_out, ["branch", "reduce"])
g.add_edge("branch", "reduce")           # fan-IN barrier: reduce waits for all
'''


# =========================================================================== #
# PATTERN 5 — HUMAN-IN-THE-LOOP (checkpointer + interrupt)                    #
# =========================================================================== #
class HITLState(TypedDict):
    proposed_path: Optional[list]
    approved: bool
    feedback: str
    best_path: Optional[list]
    best_cost: Optional[float]
    attempts: int


def build_human_in_the_loop(llm, env: GraphEnv, log: list, opts: dict | None = None):
    tools = make_tools(env, log)
    model = llm.bind_tools(tools)

    def propose(state: HITLState) -> dict:
        fb = state.get("feedback", "")
        extra = "" if not fb else (
            f"A HUMAN reviewer rejected your last path: '{fb}'. Propose a DIFFERENT "
            f"path that addresses the feedback.")
        transcript = run_react(model, tools, explorer_messages(env, extra=extra),
                               max_steps=14, on_step=record_thoughts(log))
        path, _ = extract_best_submission(transcript)
        return {"proposed_path": path, "attempts": state.get("attempts", 0) + 1}

    def review(state: HITLState) -> dict:
        # interrupt() PAUSES the graph and surfaces this payload to the app. When
        # the app resumes with Command(resume=<decision>), that value is returned
        # here. Requires a checkpointer + a thread_id in the run config.
        decision = interrupt({"proposed_path": state.get("proposed_path"),
                              "attempt": state.get("attempts")})
        return {"approved": bool(decision.get("approved")),
                "feedback": decision.get("feedback", "")}

    def finalize(state: HITLState) -> dict:
        path = state.get("proposed_path")
        ok, cost, _ = env.validate_path(path) if path else (False, None, "")
        return {"best_path": path if ok else None, "best_cost": cost if ok else None}

    def route(state: HITLState) -> str:
        if state.get("approved") or state.get("attempts", 0) >= 3:
            return "finalize"
        return "propose"

    g = StateGraph(HITLState)
    g.add_node("propose", propose)
    g.add_node("review", review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "propose")
    g.add_edge("propose", "review")
    g.add_conditional_edges("review", route, {"propose": "propose", "finalize": "finalize"})
    g.add_edge("finalize", END)
    # A CHECKPOINTER is REQUIRED for interrupt/resume — it persists state at the
    # pause so a later request can pick up exactly where it left off.
    return g.compile(checkpointer=MemorySaver())


def init_human_in_the_loop(env: GraphEnv) -> dict:
    return {"proposed_path": None, "approved": False, "feedback": "",
            "best_path": None, "best_cost": None, "attempts": 0}


def dot_human_in_the_loop(env: GraphEnv) -> str:
    return """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
  START [shape=circle, label="", width=0.2, style=filled, fillcolor=black];
  END   [shape=doublecircle, label="", width=0.2];
  propose  [label="propose\\n(agent finds a path)"];
  review   [label="review\\n(interrupt -> human)", style="rounded,filled", fillcolor="#ffd7d7"];
  finalize [label="finalize\\n(validate + submit)", style="rounded,filled", fillcolor="#d7f0d7"];
  START -> propose -> review;
  review -> propose  [label="reject +\\nfeedback", style=dashed];
  review -> finalize [label="approve", style=dashed];
  finalize -> END;
}"""


NOTES_HITL = """
**What it is.** The agent proposes a path, then **pauses for a human** to approve
or reject before it's finalized. Rejection sends feedback back to the agent, which
tries again. This is the agentic pattern for anything risky or irreversible.

**Two concepts at once**
- **Checkpointer (memory/persistence).** `compile(checkpointer=MemorySaver())` saves
  the graph's state at each step. That's what lets a run *pause* and *resume* across
  separate requests (and, more broadly, gives agents durable memory).
- **`interrupt()` + `Command(resume=...)`.** `interrupt(payload)` stops the graph and
  hands `payload` to your app. You show it to a human, then call
  `graph.invoke(Command(resume=decision), config)` — execution continues from the
  interrupt, with `decision` as the return value of `interrupt()`.

**The run config matters.** interrupt/resume needs a stable `thread_id`:
`config = {"configurable": {"thread_id": "run-1"}}`. The checkpointer keys state by
that id, so the resume finds the paused state.

**Gotchas**
- Keep the SAME compiled graph object between pause and resume (its MemorySaver holds
  the state). In Streamlit, stash it in `st.session_state`.
- Cap retries — a human might keep rejecting; the graph should still terminate.
"""

SNIPPET_HITL = '''from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def review(state):
    decision = interrupt({"proposed_path": state["proposed_path"]})  # PAUSE here
    return {"approved": decision["approved"], "feedback": decision.get("feedback", "")}

graph = g.compile(checkpointer=MemorySaver())        # checkpointer REQUIRED
config = {"configurable": {"thread_id": "run-1"}}    # keys the saved state

# 1) run until it pauses at interrupt()
for chunk in graph.stream(init_state, config):
    if "__interrupt__" in chunk:
        payload = chunk["__interrupt__"][0].value    # {"proposed_path": [...]}

# 2) a human decides... then resume from exactly where it paused
graph.invoke(Command(resume={"approved": False, "feedback": "avoid node D"}), config)
'''


# --------------------------------------------------------------------------- #
# REGISTRY                                                                     #
# --------------------------------------------------------------------------- #
PATTERNS: dict[str, Pattern] = {
    "single_react": Pattern(
        key="single_react", title="1 · Single ReAct Agent",
        tagline="One agent, one loop: reason -> act -> observe -> repeat.",
        concepts=["StateGraph", "add_messages reducer", "ToolNode",
                  "conditional_edges", "the ReAct loop"],
        theory="The **atom** of agentic AI. One LLM node reasons and may request a "
               "tool; a `ToolNode` executes it; control loops back so the model can "
               "read the result and think again — **ReAct** (reason → act → observe). "
               "This tiny loop *is* the definition of an 'agent'; every heavier "
               "framework is sugar over it. It shines on **discovery** tasks like this "
               "one, where the model starts blind and must *earn* the information by "
               "exploring. The one non-negotiable is a **step cap** — an unbounded "
               "loop is a runaway-cost incident waiting to happen.",
        notes=NOTES_REACT, snippet=SNIPPET_REACT,
        build=build_single_react, init=init_single_react, dot=dot_single_react),
    "supervisor": Pattern(
        key="supervisor", title="2 · Supervisor (Orchestrator–Worker)",
        tagline="A router delegates to specialist worker agents and back.",
        concepts=["multi-agent handoff", "routing node", "shared-state channels",
                  "LLM-proposes / harness-disposes", "code agents vs LLM agents"],
        theory="An **orchestrator** node decides which specialist worker runs next, "
               "loops through them, and stops — the shape behind most 'multi-agent' "
               "systems. Here a greedy explorer (LLM), a cautious explorer (LLM), and "
               "a **deterministic verifier** (pure code) never call each other; they "
               "coordinate through **shared-state channels** merged by reducers (loose "
               "coupling — add a worker without touching the rest). The lesson is "
               "**LLM-proposes / harness-disposes**: the router *asks* the model which "
               "worker to run, but the harness enforces a legal, terminating order — "
               "the central tension of agentic design, autonomy vs. control.",
        notes=NOTES_SUPERVISOR, snippet=SNIPPET_SUPERVISOR,
        build=build_supervisor, init=init_supervisor, dot=dot_supervisor),
    "evaluator_optimizer": Pattern(
        key="evaluator_optimizer",
        title="3 · Evaluator–Optimizer (Reflection + Context Shrinking)",
        tagline="Propose -> critique -> improve, with the context window managed each loop.",
        concepts=["reflection / self-correction", "cyclic graph + loop guard",
                  "ground-truth stop", "context trimming vs summarizing",
                  "token accounting"],
        theory="A **reflection loop**: an optimizer proposes a path, an evaluator "
               "critiques it and decides stop-or-retry. Self-correction is what lifts "
               "an agent above one-shot prompting. Two stop signals on purpose — "
               "**ground truth** (is it optimal?) and a **loop guard** (max "
               "iterations), because you never trust a model to always converge. The "
               "headline lesson rides along: every loop appends messages, so between "
               "iterations the history is **trimmed or summarized** to keep the "
               "context window bounded — switch the strategy and watch the token "
               "trace change, before vs after.",
        notes=NOTES_EVAL, snippet=SNIPPET_EVAL,
        build=build_evaluator_optimizer, init=init_evaluator_optimizer,
        dot=dot_evaluator_optimizer),
    "parallel_explorers": Pattern(
        key="parallel_explorers", title="4 · Parallel Explorers (Map–Reduce with Send)",
        tagline="Fan out one agent per first-hop, run in parallel, reduce to the best.",
        concepts=["Send API (dynamic fan-out)", "parallel superstep",
                  "reducer-based aggregation", "fan-in barrier"],
        theory="**Map-reduce for agents.** `Send` dynamically fans out one explorer "
               "per first-hop neighbour of the source; they run in **parallel**, then "
               "**fan in** to a `reduce` node that merges their results and keeps the "
               "cheapest. The number of branches is decided at *runtime* from the "
               "state, and a reducer channel absorbs the concurrent writes without a "
               "race. It's the pattern for independent sub-problems (beam search, "
               "multi-query RAG, ensembling) — powerful, but N branches cost N× the "
               "tokens, so pair it with pruning when the fan-out is wide.",
        notes=NOTES_PARALLEL, snippet=SNIPPET_PARALLEL,
        build=build_parallel_explorers, init=init_parallel_explorers,
        dot=dot_parallel_explorers),
    "human_in_the_loop": Pattern(
        key="human_in_the_loop", title="5 · Human-in-the-Loop (Checkpointer + interrupt)",
        tagline="The agent pauses for your approval before finalizing; you can reject with feedback.",
        concepts=["checkpointer / persistence", "interrupt() + Command(resume)",
                  "thread_id config", "durable memory", "control over autonomy"],
        theory="The agent proposes a path and then **pauses via `interrupt()`** for a "
               "human to approve or reject before anything is finalized — the pattern "
               "for risky or irreversible actions. It stacks two concepts: a "
               "**checkpointer** persists the graph's state at the pause (durable "
               "memory), and `interrupt()` + `Command(resume=…)` hand control to your "
               "app and back, resuming *exactly* where it stopped. Rejection feeds "
               "your feedback into the next attempt, so control and autonomy compose "
               "instead of competing — with a retry cap so a stubborn reviewer can't "
               "loop forever.",
        notes=NOTES_HITL, snippet=SNIPPET_HITL,
        build=build_human_in_the_loop, init=init_human_in_the_loop,
        dot=dot_human_in_the_loop, interactive=True),
}
