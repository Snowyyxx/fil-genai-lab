"""Headless run of the agentic shortest-path app — no Streamlit UI.

Runs a pattern end-to-end against a real local LLM (Ollama), streams the node
activity, then scores the agent's answer against the Dijkstra ground truth.

Usage:
    .venv/bin/python run_headless.py                 # single_react on a 6-node graph
    .venv/bin/python run_headless.py supervisor 5    # pattern + graph size
"""
from __future__ import annotations

import sys

from graph_env import random_graph
from llm import get_llm
from patterns import PATTERNS
from agent_toolkit import extract_best_submission

pattern_key = sys.argv[1] if len(sys.argv) > 1 else "single_react"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6

env = random_graph(n=n, density=0.5, seed=7)
print("GRAPH:"); print(env.to_text())
opt_cost, opt_path = env.dijkstra()
print(f"\nGROUND TRUTH (Dijkstra): {opt_path}  cost={opt_cost}\n")

llm = get_llm(provider="ollama", model="qwen2.5:3b-instruct", temperature=0.0)
pat = PATTERNS[pattern_key]
log: list = []
graph = pat.build(llm, env, log)
state = pat.init(env)

print(f"=== running pattern: {pat.title} ===")
config = {"recursion_limit": 60}
final = None
for chunk in graph.stream(state, config=config, stream_mode="values"):
    final = chunk

# pull the agent's best answer back out of the transcript / state
best_path = final.get("best_path") if isinstance(final, dict) else None
best_cost = final.get("best_cost") if isinstance(final, dict) else None
if best_path is None and isinstance(final, dict) and "messages" in final:
    best_path, best_cost = extract_best_submission(final["messages"])

tool_calls = [e for e in log if e["tool"] in ("get_neighbors", "submit_path")]
print("\n=== TOOL CALLS (the agent exploring) ===")
for e in tool_calls:
    print(f"  {e['tool']}({e['args']}) -> {e['result']}")

print("\n=== RESULT ===")
print(f"  agent path : {best_path}  cost={best_cost}")
print(f"  optimum    : {opt_path}  cost={opt_cost}")
print(f"  tool calls : {len(tool_calls)}")
verdict = "OPTIMAL ✓" if best_cost == opt_cost else (
    "valid but sub-optimal" if best_cost else "no valid path found")
print(f"  verdict    : {verdict}")
