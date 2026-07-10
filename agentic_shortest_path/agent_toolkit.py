"""agent_toolkit.py — the reusable building blocks every pattern shares.

This is the "harness engineering" layer. An LLM on its own is just a text
predictor; the harness is everything around it that turns it into a dependable
agent:

  * TOOLS          — typed functions the model may call (its hands)
  * a RUN LOOP     — model -> tool -> model -> ... until it stops (the ReAct loop)
  * GUARDRAILS     — step caps, verification, structured extraction
  * CONTEXT MGMT   — trimming old messages so we don't overflow the context window
  * OBSERVABILITY  — a shared log so the UI can show every tool call

Read this file top-to-bottom; the four orchestration patterns in patterns.py are
just different ways of *wiring these blocks together*.
"""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import tool

from graph_env import GraphEnv


# --------------------------------------------------------------------------- #
# 1. TOOLS — the agent's only way to touch the world.                          #
#    We build them with a factory so each tool "closes over" a specific graph  #
#    environment and a shared `log` list (for the UI). The docstrings matter:  #
#    the model reads them to decide when/how to call each tool.                #
# --------------------------------------------------------------------------- #
def make_tools(env: GraphEnv, log: list[dict] | None = None) -> list:
    def _record(name: str, args: dict, result: str) -> None:
        if log is not None:
            log.append({"tool": name, "args": args, "result": result})

    @tool
    def get_neighbors(node: str) -> str:
        """Look up the outgoing edges from `node`. Returns 'neighbor:cost' pairs
        (e.g. 'B:3, C:7'), or '(no outgoing edges)'. This is how you explore the
        graph one step at a time — call it on your current node to see options."""
        nbrs = env.neighbors(node)
        out = ", ".join(f"{v}:{int(w)}" for v, w in nbrs) or "(no outgoing edges)"
        _record("get_neighbors", {"node": node}, out)
        return out

    @tool
    def submit_path(path: str) -> str:
        """Submit a full candidate route as comma-separated nodes, e.g. 'A,C,E'.
        Returns whether it is valid and its exact total cost. ALWAYS finish by
        calling this — never state a cost you computed yourself."""
        nodes = [p.strip() for p in path.split(",") if p.strip()]
        ok, cost, note = env.validate_path(nodes)
        out = f"valid={ok} cost={cost} note={note}"
        _record("submit_path", {"path": nodes}, out)
        return out

    return [get_neighbors, submit_path]


# --------------------------------------------------------------------------- #
# 2. PROMPTS — the agent's job description. Prompting IS harness engineering.   #
# --------------------------------------------------------------------------- #
EXPLORER_SYSTEM = """You are a routing agent on a weighted DIRECTED graph.
Goal: find the LOWEST total-cost path from {source} to {target}.

You cannot see the whole graph. Discover it with tools:
- get_neighbors(node): the outgoing edges of `node` as 'neighbor:cost' pairs.
- submit_path('{source},...,{target}'): validate a full path and get its true cost.

Search method: DEPTH-FIRST SEARCH with BACKTRACKING and cost-based pruning
(branch-and-bound). Think out loud and keep an explicit CURRENT PATH (a stack of
nodes) that starts at {source}, its running cost, the cheapest complete valid path
found so far (call it BEST — its cost starts at infinity), and, for each node, which
neighbours you have already tried.
1. EXPAND: at the deepest node of the current path, call get_neighbors to list its
   outgoing edges (its cost options).
2. GO DEEPER: choose ONE neighbour that is not already on the current path, append it
   to the path, add the edge cost, and repeat from step 1 at that neighbour.
3. PRUNE (branch-and-bound): whenever the running cost of the current path is >= BEST,
   this branch cannot beat what you already have — abandon it and BACKTRACK at once,
   WITHOUT expanding further. Check this after every step you add.
4. REACH GOAL: when the deepest node is {target}, you have one complete candidate —
   call submit_path to get its TRUE cost; if it is cheaper than BEST, it becomes the
   new BEST (a tighter bound makes future pruning cut more).
5. BACKTRACK: when you prune (step 3), hit a dead end, or have tried every neighbour
   of the deepest node, pop it off the path, subtract its edge cost, return to the
   previous node, and try its NEXT untried neighbour. Announce each one, e.g.
   "cost 19 >= best 17, prune and backtrack to B" or "dead end at D, backtrack to B".
6. EFFICIENCY: do NOT call get_neighbors twice on the same node — once you have its
   edges, reuse them from your notes. This is a SMALL graph; you should finish within
   about a dozen tool calls.
7. STOP: don't settle for the very first path before checking the alternatives, but
   once every branch out of {source} has reached {target} or been pruned/dead-ended
   (no untried path can beat BEST), state the CHEAPEST valid path (BEST) and its total
   cost in ONE sentence and STOP. Never re-explore a path you have already evaluated.

{extra}"""


def explorer_messages(env: GraphEnv, extra: str = "") -> list[BaseMessage]:
    """Standard opening messages for an exploration agent."""
    sys = SystemMessage(
        content=EXPLORER_SYSTEM.format(source=env.source, target=env.target, extra=extra)
    )
    human = HumanMessage(
        content=f"Find the cheapest path from {env.source} to {env.target} using "
        f"depth-first search with backtracking. Begin your DFS at {env.source}."
    )
    return [sys, human]


# --------------------------------------------------------------------------- #
# 3. THE REACT LOOP — the atom of agentic behavior.                            #
#    "ReAct" = Reason + Act: the model thinks, calls a tool, reads the result, #
#    thinks again... This tiny loop is literally what an "agent" is. Every      #
#    prebuilt agent (LangGraph's create_react_agent, etc.) is a fancier version #
#    of this. We hand-roll it so the mechanism is not a black box.             #
# --------------------------------------------------------------------------- #
def run_react(
    model_with_tools,
    tools: list,
    messages: list[BaseMessage],
    max_steps: int = 14,
    on_step: Callable[[BaseMessage], None] | None = None,
) -> list[BaseMessage]:
    """Drive one agent to completion. Returns the full message transcript.

    max_steps is a GUARDRAIL: without it a confused model can loop forever and
    burn tokens. Real harnesses always cap the loop.
    """
    tools_by_name = {t.name: t for t in tools}
    convo = list(messages)
    for _ in range(max_steps):
        ai: AIMessage = model_with_tools.invoke(convo)
        convo.append(ai)
        if on_step:
            on_step(ai)
        if not getattr(ai, "tool_calls", None):
            break  # no tool call -> the agent is done talking
        for call in ai.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            tmsg = ToolMessage(content=str(result), tool_call_id=call["id"])
            convo.append(tmsg)
            if on_step:
                on_step(tmsg)
    return convo


def record_thoughts(log: list[dict] | None) -> Callable[[BaseMessage], None]:
    """Return an `on_step` callback that appends the model's REASONING to the shared
    log as it happens. Patterns that run a full ReAct loop *inside* one graph node
    (supervisor, parallel, HITL) don't stream their internal messages to the UI, so
    without this the model's chain-of-thought would be invisible. We tag these entries
    `tool="think"` so the UI can render them as 💭 and keep them out of the tool table.
    """
    def on_step(msg: BaseMessage) -> None:
        if log is None or not isinstance(msg, AIMessage):
            return
        text = str(getattr(msg, "content", "")).strip()
        if text:
            log.append({"tool": "think", "args": {}, "result": text})
    return on_step


def extract_best_submission(messages: list[BaseMessage]) -> tuple[list[str] | None, float | None]:
    """Scan a transcript for submit_path calls and return the cheapest VALID one.

    Agents don't return clean data — they return chat. Pulling structured facts
    back out of the transcript is a routine harness chore. We pair each
    submit_path tool call with its ToolMessage result to read validity + cost.
    """
    best_path: list[str] | None = None
    best_cost: float | None = None
    pending: dict[str, list[str]] = {}  # tool_call_id -> submitted node list
    for m in messages:
        if isinstance(m, AIMessage):
            for call in getattr(m, "tool_calls", []) or []:
                if call["name"] == "submit_path":
                    raw = str(call["args"].get("path", ""))
                    pending[call["id"]] = [p.strip() for p in raw.split(",") if p.strip()]
        elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
            text = str(m.content)
            if "valid=True" in text:
                # parse "cost=<n>"
                try:
                    cost = float(text.split("cost=")[1].split()[0])
                except (IndexError, ValueError):
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_path = pending[m.tool_call_id]
    return best_path, best_cost


# --------------------------------------------------------------------------- #
# 4. CONTEXT-WINDOW SHRINKING.                                                 #
#    Every model has a fixed context window (e.g. 8k–32k tokens). A long agent #
#    run piles up messages until it overflows — costs rise, latency rises, and #
#    eventually the request fails. The fix is to *shrink* the message history   #
#    while keeping what matters. Two standard moves:                           #
#      (a) TRIM   — keep the system prompt + the most recent N tokens.         #
#      (b) SUMMARIZE — replace old turns with a short LLM-written summary.      #
#    Below is trim (deterministic, cheap). The evaluator/optimizer pattern uses #
#    it between iterations; the UI shows the before/after token estimate.       #
# --------------------------------------------------------------------------- #
def estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token count (~4 chars/token). Good enough to *show* the shrink.
    For real budgeting use the model's own tokenizer / count_tokens endpoint."""
    chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    return chars // 4


def trim_history(
    messages: list[BaseMessage], max_tokens: int = 400
) -> list[BaseMessage]:
    """Keep the system message + the most recent messages under a token budget.

    Uses LangChain's trim_messages with our char-based estimator so it runs
    without a real tokenizer. `include_system=True` protects the job description;
    `start_on='human'` keeps the surviving history conversational and valid.
    """
    return trim_messages(
        messages,
        token_counter=lambda msgs: estimate_tokens(msgs),
        max_tokens=max_tokens,
        strategy="last",
        include_system=True,
        start_on="human",
        allow_partial=False,
    )


def summarize_history(llm, messages: list[BaseMessage], keep_last: int = 2) -> list[BaseMessage]:
    """SUMMARIZE old messages into one recap, keep the last few verbatim.

    Trimming *drops* old turns; summarizing *compresses* them, so the agent keeps
    the meaning of what happened (which node it explored, costs it found) at the
    price of one extra LLM call. Use it when early context still matters.
    """
    if len(messages) <= keep_last + 1:
        return messages
    lead: list[BaseMessage] = []
    body = list(messages)
    if body and isinstance(body[0], SystemMessage):
        lead, body = [body[0]], body[1:]           # protect the job description
    tail = body[-keep_last:] if keep_last else []
    head = body[:-keep_last] if keep_last else body
    if not head:
        return messages
    transcript = "\n".join(
        f"{type(m).__name__}: {str(getattr(m, 'content', ''))[:300]}" for m in head
    )
    recap = llm.invoke([
        SystemMessage(content="Summarize the agent's progress in 3-4 terse bullets: "
                      "nodes explored, edge costs seen, best path/cost so far. This "
                      "replaces the raw history, so keep every fact that matters."),
        HumanMessage(content=transcript),
    ])
    summary = SystemMessage(content="Summary of earlier steps:\n" + str(recap.content))
    return lead + [summary] + tail
