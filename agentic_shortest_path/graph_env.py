"""graph_env.py — the *world* the agents act in.

Agentic AI 101: an agent needs an ENVIRONMENT it can only touch through actions
(tools). Here the environment is a weighted directed graph. The agents never get
the whole graph handed to them — they must *discover* it by calling
`neighbors(node)`, exactly like a delivery robot that can only see the roads at
its current intersection.

We keep `dijkstra()` here too, but that is the *answer key* — it belongs to the
harness (our evaluation code), NOT to the agents. Keeping ground truth on the
environment lets us score whether the agent actually found the optimum.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass


@dataclass
class GraphEnv:
    """A weighted directed graph plus a source and target node.

    `adjacency[u][v] = w` means there is an edge u -> v with cost w.
    Edges point "forward" (A->B->C...) so the demo graphs are acyclic and every
    exploration terminates — one less thing to fight while learning agents.
    """

    adjacency: dict[str, dict[str, float]]
    source: str
    target: str

    # ---- the read-only "senses" the agents get, wrapped as tools elsewhere ----
    def nodes(self) -> list[str]:
        return list(self.adjacency)

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        """Outgoing edges from `node`, sorted. The agent's only way to see ahead."""
        return sorted(self.adjacency.get(node, {}).items())

    def edge_cost(self, u: str, v: str) -> float | None:
        return self.adjacency.get(u, {}).get(v)

    def path_cost(self, path: list[str]) -> float | None:
        """Total cost of a node list, or None if any edge is missing."""
        total = 0.0
        for u, v in zip(path, path[1:]):
            w = self.edge_cost(u, v)
            if w is None:
                return None
            total += w
        return total

    def validate_path(self, path: list[str]) -> tuple[bool, float | None, str]:
        """Harness-grade validator used by the `submit_path` tool.

        Returns (is_valid, cost, note). This is the deterministic ground truth an
        LLM must never be trusted to compute itself.
        """
        if not path:
            return False, None, "empty path"
        if path[0] != self.source:
            return False, None, f"path must start at source '{self.source}'"
        if path[-1] != self.target:
            return False, None, f"path must end at target '{self.target}'"
        cost = self.path_cost(path)
        if cost is None:
            return False, None, "path uses an edge that does not exist"
        if len(set(path)) != len(path):
            return False, cost, "path revisits a node (not a simple path)"
        return True, cost, "ok"

    # ---- ground truth: the harness/verifier uses this, the LLM never sees it ----
    def dijkstra(self) -> tuple[float | None, list[str] | None]:
        """Optimal cost and path, for scoring the agents."""
        pq: list[tuple[float, str, list[str]]] = [(0.0, self.source, [self.source])]
        best: dict[str, float] = {self.source: 0.0}
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == self.target:
                return cost, path
            for nbr, w in self.adjacency.get(node, {}).items():
                nc = cost + w
                if nc < best.get(nbr, float("inf")):
                    best[nbr] = nc
                    heapq.heappush(pq, (nc, nbr, path + [nbr]))
        return None, None

    def to_text(self) -> str:
        """A compact human/LLM-readable dump (used only for teaching displays)."""
        lines = [f"source={self.source} target={self.target}"]
        for u in self.nodes():
            edges = ", ".join(f"{v}:{w}" for v, w in self.neighbors(u)) or "(sink)"
            lines.append(f"  {u} -> {edges}")
        return "\n".join(lines)


def random_graph(
    n: int = 6, density: float = 0.5, max_w: int = 9, seed: int = 0
) -> GraphEnv:
    """Build a reproducible random forward-directed graph on nodes A, B, C, ...

    A "spine" edge between consecutive nodes guarantees the target is reachable,
    so every run has an answer. Forward-only edges keep it acyclic.
    """
    rnd = random.Random(seed)
    names = [chr(ord("A") + i) for i in range(n)]
    adj: dict[str, dict[str, float]] = {x: {} for x in names}
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            if rnd.random() < density:
                adj[u][v] = rnd.randint(1, max_w)
    # guarantee a source->target route via the spine A->B->C->...->last
    for u, v in zip(names, names[1:]):
        adj[u].setdefault(v, rnd.randint(1, max_w))
    return GraphEnv(adjacency=adj, source=names[0], target=names[-1])


if __name__ == "__main__":
    env = random_graph(n=6, density=0.4, seed=7)
    print(env.to_text())
    print("optimum:", env.dijkstra())
