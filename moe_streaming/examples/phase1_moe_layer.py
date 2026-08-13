"""
PHASE 1 — the heart of the article, in isolation.
=================================================

The whole "run a giant model on tiny RAM" trick lives in ONE place: the
Mixture-of-Experts (MoE) layer. This file rebuilds *just that layer*, with no
attention / tokenizer / transformer machinery to distract you.

The idea
--------
An MoE layer has N "experts" (each expert is a small feed-forward network).
For every input, a tiny "router" scores the experts and picks only the best
TOP_K of them. So even with N=128 experts, each input only ever uses 2.

That sparsity is the opening. If we:
  1. store all experts CONTIGUOUSLY in one file on disk, and
  2. `memory-map` that file (map it into our address space WITHOUT loading it),
then when we read only the 2 selected experts, the OS pages in only those 2
experts' bytes. RAM stays tiny no matter how big the file is.

This script proves it: it builds a ~1 GB file of experts, then shows that
- "streaming" (read only selected experts) grows RAM by a few MB, while
- "naive" (load the whole file) grows RAM by ~1 GB.

Run it:  python phase1_moe_layer.py
"""
from __future__ import annotations

import os

import numpy as np

# ── Config (a toy MoE layer) ────────────────────────────────────────────────
DIM = 512            # size of one token's vector
HIDDEN = 2048        # hidden width inside each expert's FFN
N_EXPERTS = 128      # total experts — "the whole library"
TOP_K = 2            # experts actually used per token — "the sparsity"
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "experts.bin")

DTYPE = np.float32
# Each expert is a 2-layer FFN: W1 (DIM×HIDDEN) then W2 (HIDDEN×DIM).
EXPERT_FLOATS = DIM * HIDDEN + HIDDEN * DIM
EXPERT_BYTES = EXPERT_FLOATS * 4


# ── Step 1: the "exporter" — write all experts contiguously to disk ──────────
def build_experts_on_disk() -> None:
    """Mimics the offline step that converts model weights into our own binary
    format. Experts are written back-to-back so each is a contiguous blob —
    THIS LAYOUT is what makes memory-mapped streaming efficient."""
    if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) == N_EXPERTS * EXPERT_BYTES:
        return
    rng = np.random.default_rng(0)
    with open(WEIGHTS_PATH, "wb") as f:
        for _ in range(N_EXPERTS):
            w1 = (rng.standard_normal((DIM, HIDDEN)) * 0.02).astype(DTYPE)
            w2 = (rng.standard_normal((HIDDEN, DIM)) * 0.02).astype(DTYPE)
            f.write(w1.tobytes())
            f.write(w2.tobytes())


# ── Helpers ──────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    """Resident memory (actually-in-RAM) of this process, in MB. Linux only."""
    pages = int(open("/proc/self/statm").read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def read_expert(mm: np.memmap, e: int):
    """Slice expert `e`'s two weight matrices out of the flat memmap.
    Accessing these slices is what triggers the OS to page in ONLY expert e."""
    base = e * EXPERT_FLOATS
    w1 = mm[base : base + DIM * HIDDEN].reshape(DIM, HIDDEN)
    w2 = mm[base + DIM * HIDDEN : base + EXPERT_FLOATS].reshape(HIDDEN, DIM)
    return w1, w2


def expert_ffn(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """One expert's feed-forward network:  x -> ReLU(x·W1) -> ·W2 ."""
    return np.maximum(x @ w1, 0.0) @ w2


def router(x: np.ndarray, gate: np.ndarray):
    """The tiny network that decides WHICH experts to use.
    Scores all experts, keeps the top-k, softmax-normalizes their weights."""
    scores = x @ gate                       # (N_EXPERTS,)
    top = np.argsort(-scores)[:TOP_K]       # indices of the best TOP_K experts
    w = np.exp(scores[top] - scores[top].max())
    w /= w.sum()                            # how much each chosen expert contributes
    return top, w


# ── Main: compare streaming vs. naive ────────────────────────────────────────
def main() -> None:
    build_experts_on_disk()
    file_mb = N_EXPERTS * EXPERT_BYTES / 1e6
    print(f"Model on disk : {file_mb:.0f} MB  ({N_EXPERTS} experts, {EXPERT_BYTES/1e6:.1f} MB each)")
    print(f"Sparsity      : top_k = {TOP_K}  → each token uses {TOP_K}/{N_EXPERTS} experts\n")

    rng = np.random.default_rng(1)
    x = (rng.standard_normal(DIM) * 0.1).astype(DTYPE)          # a token vector
    gate = (rng.standard_normal((DIM, N_EXPERTS)) * 0.02).astype(DTYPE)  # router weights

    # ---- STREAMING: memory-map the file, read only the selected experts ----
    before = rss_mb()
    mm = np.memmap(WEIGHTS_PATH, dtype=DTYPE, mode="r")   # maps file; loads NOTHING yet
    top, weights = router(x, gate)
    out = np.zeros(DIM, dtype=DTYPE)
    for e, contribution in zip(top, weights):
        w1, w2 = read_expert(mm, int(e))                 # ← touches only expert e's pages
        out += contribution * expert_ffn(x, w1, w2)
    stream_growth = rss_mb() - before
    read_mb = TOP_K * EXPERT_BYTES / 1e6
    print(f"[STREAMING]  used experts {list(map(int, top))}")
    print(f"             read ~{read_mb:.0f} MB of {file_mb:.0f} MB on disk")
    print(f"             RAM grew ~{stream_growth:.1f} MB   ← stays tiny\n")

    # ---- NAIVE: load the whole file into RAM (what a normal loader does) ----
    before2 = rss_mb()
    all_weights = np.fromfile(WEIGHTS_PATH, dtype=DTYPE)  # pulls the ENTIRE file into RAM
    naive_growth = rss_mb() - before2
    print(f"[NAIVE]      loaded ALL {N_EXPERTS} experts")
    print(f"             RAM grew ~{naive_growth:.0f} MB   ← this is what OOM-kills you")
    del all_weights

    print(f"\nTakeaway: same model, same output — but streaming used "
          f"~{naive_growth/max(stream_growth,0.1):.0f}x less RAM by reading only "
          f"the {TOP_K} experts it needed. Scale N_EXPERTS up and the gap grows without bound.")


if __name__ == "__main__":
    main()
