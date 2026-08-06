"""Render the Faraz Shayari RAG system-design diagram to a PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette -------------------------------------------------------------
INK   = "#0f172a"; MUTE = "#475569"
SRC   = ("#e5e7eb", "#6b7280")   # external source (gray)
CODE  = ("#dbeafe", "#2563eb")   # our python code (blue)
DATA  = ("#fef3c7", "#d97706")   # data artifact / file (amber)
AWS   = ("#ffedd5", "#ea580c")   # Amazon Bedrock managed (orange)
USER  = ("#ede9fe", "#7c3aed")   # user (purple)
OUT   = ("#d1fae5", "#059669")   # response (green)
LANE_OFF = "#fafaf5"; LANE_ON = "#f5fbfb"

fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")


def box(x, y, w, h, title, sub, colors, tsize=11, ssize=8.3):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=3))
    cx = x + w / 2
    ax.text(cx, y + h - 0.30, title, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(cx, y + 0.28, sub, ha="center", va="center",
                fontsize=ssize, color=MUTE, zorder=4, wrap=True)
    return {"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": y + h / 2}


def arrow(a, b, side="RL", color="#334155", ls="-", lw=1.9, rad=0.0, label=None):
    pts = {"R": (a["x"] + a["w"], a["cy"]), "L": (a["x"], a["cy"]),
           "T": (a["cx"], a["y"] + a["h"]), "B": (a["cx"], a["y"])}
    x1, y1 = pts[side[0]]
    x2, y2 = {"R": (b["x"] + b["w"], b["cy"]), "L": (b["x"], b["cy"]),
              "T": (b["cx"], b["y"] + b["h"]), "B": (b["cx"], b["y"])}[side[1]]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color, linestyle=ls, zorder=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center",
                fontsize=7.6, color=MUTE, style="italic", zorder=5)


# ---- title ---------------------------------------------------------------
ax.text(8, 10.62, "Faraz Shayari — RAG System Architecture", ha="center",
        fontsize=20, fontweight="bold", color=INK)
ax.text(8, 10.18, "Situation in → a new shayari in Ahmad Faraz's voice, grounded in his real couplets  "
        "·  LiteLLM + Amazon Bedrock + ECS Fargate",
        ha="center", fontsize=10.5, color=MUTE)

# ==== OFFLINE lane ========================================================
ax.add_patch(FancyBboxPatch((0.4, 7.55), 15.2, 2.3, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor="#cbd5e1", facecolor=LANE_OFF, zorder=1))
ax.text(0.62, 9.66, "①  OFFLINE  —  build the index (run once, on a laptop)",
        ha="left", fontsize=11.5, fontweight="bold", color="#b45309")

ox, ow, oy, oh = 0.75, 2.55, 8.0, 1.15
gap = (14.5 - 5 * ow) / 4
xs = [ox + i * (ow + gap) for i in range(5)]
a1 = box(xs[0], oy, ow, oh, "Rekhta dataset", "GitHub urdu_ghazals_rekhta\n50 Faraz ghazals (Roman Urdu)", SRC, 10.5)
a2 = box(xs[1], oy, ow, oh, "parse_corpus.py", "split ghazals into\nself-contained couplets", CODE, 10.5)
a3 = box(xs[2], oy, ow, oh, "couplets.jsonl", "374 couplets\n(sher = retrieval unit)", DATA, 10.5)
a4 = box(xs[3], oy, ow, oh, "Titan Embed v2", "Amazon Bedrock\nembed each → 1024-d vector", AWS, 10.5)
a5 = box(xs[4], oy, ow, oh, "embeddings.npy", "374 × 1024 index\n(L2-normalized)", DATA, 10.5)
for a, b in [(a1, a2), (a2, a3), (a3, a4), (a4, a5)]:
    arrow(a, b, "RL")

# ==== ONLINE lane =========================================================
ax.add_patch(FancyBboxPatch((0.4, 0.75), 15.2, 6.5, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.3, edgecolor="#cbd5e1", facecolor=LANE_ON, zorder=1))
ax.text(0.62, 7.02, "②  ONLINE  —  per request  (runs inside the ECS Fargate container; "
        "Bedrock reached via the IAM task role — no keys in the image)",
        ha="left", fontsize=11.5, fontweight="bold", color="#0f766e")

# row 1 (left → right)
r1y, bw, bh = 5.05, 3.0, 1.25
r1x = [0.9, 4.6, 8.3, 12.0]
b1 = box(r1x[0], r1y, bw, bh, "User / Browser", "types a life situation\n(English or Roman Urdu)", USER, 11)
b2 = box(r1x[1], r1y, bw, bh, "FastAPI app", "GET / (chat UI)\nPOST /shayari", CODE, 11)
b3 = box(r1x[2], r1y, bw, bh, "Embed situation", "Titan v2 · Bedrock\nsame model as corpus", AWS, 11)
b4 = box(r1x[3], r1y, bw, bh, "Retrieve (cosine)", "top-k nearest\nreal couplets", CODE, 11)
arrow(b1, b2, "RL", label="situation"); arrow(b2, b3, "RL"); arrow(b3, b4, "RL", label="query vector")

# index feeds retrieval (cross-lane dashed)
arrow(a5, b4, "BT", color="#d97706", ls=(0, (4, 3)), lw=1.7, rad=0.0, label="index loaded at startup")

# row 2 (right → left visual flow, drawn left→right boxes)
r2y = 2.35
r2x = [0.9, 4.6, 8.3, 12.0]
b5 = box(r2x[1], r2y, bw, bh, "Prompt builder", "situation + retrieved\ncouplets as style anchors", CODE, 11)
b6 = box(r2x[2], r2y, bw, bh, "LiteLLM → Bedrock LLM", "Claude / Nova composes\na NEW shayari in his voice", AWS, 11)
b7 = box(r2x[3], r2y, bw, bh, "Response", "AI shayari + the real\ncouplets that inspired it", OUT, 11)
# connect B4 down to B5, then across
arrow(b4, b5, "BT", color="#334155", rad=-0.15, label="top-k couplets")
arrow(b5, b6, "RL"); arrow(b6, b7, "RL", label="composed shayari")
# response back to the user (up the left)
arrow(b7, b1, "TL", color="#059669", rad=-0.28, label="rendered in the UI")

# ---- legend --------------------------------------------------------------
leg = [("External source", SRC), ("Our code", CODE), ("Data artifact / index", DATA),
       ("Amazon Bedrock (managed)", AWS), ("User", USER), ("Response", OUT)]
lx = 0.9
for name, (fc, ec) in leg:
    ax.add_patch(FancyBboxPatch((lx, 0.2), 0.32, 0.26, boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(lx + 0.42, 0.33, name, ha="left", va="center", fontsize=9, color=INK)
    lx += 0.42 + 0.02 + len(name) * 0.105 + 0.5

OUTP = Path(__file__).parent / "faraz_shayari_architecture.png"
fig.savefig(OUTP, bbox_inches="tight", facecolor="white", dpi=200)
print(f"saved {OUTP}")
