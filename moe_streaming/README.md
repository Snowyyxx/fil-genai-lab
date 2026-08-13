# moe_streaming — giant models, tiny RAM (from scratch)

Building, from zero, the trick behind “run a trillion-parameter model on 8 GB”:
**Mixture-of-Experts sparsity + memory-mapped weight streaming**. A real Granite 3
MoE runs here with its experts kept **on disk** and streamed in — by our own code,
one top-k slice per token — so a ~5.8 GB model runs in **under 1 GB of RAM**.

## Run it

```bash
python moe_streaming/serve.py        # → http://localhost:8200
```

Open the page, pick **🧩 granite-moe (ours)**, and send a prompt. Watch the expert
strip light the *actual* routed experts and the panel flip between **● RESIDENT**
and **⟳ DISK STREAMING** (toggle “force reads from disk”). First send loads the
model and builds `engine/experts.bin` (~5 GB, one-time, git-ignored).

## Layout

```
moe_streaming/
├── serve.py              # entry point: serves web/, runs engine/ or proxies Ollama
├── engine/               # ★ the real from-scratch streaming engine
│   ├── engine.py         #   loads model, frees expert RAM, streams experts from disk
│   ├── disk_bank.py      #   mmap reader + fault-forcing eviction (the core trick)
│   ├── README.md         #   how it works, with measurements
│   └── experts.bin*      #   all experts, flat on disk (generated, git-ignored)
├── web/                  # browser UIs
│   ├── chat.html         #   the real chat (served at /)
│   ├── demo.html         #   interactive animated simulator
│   └── guide.html        #   illustrated explainer (transformers, MoE, mmap, C)
├── examples/
│   └── phase1_moe_layer.py   # standalone toy: 1 GB experts on disk, ~17 MB in RAM
└── docs/
    └── ROADMAP.md        # the 5-phase build plan (Python first, C last)
```

\* generated on first run.

## The idea in one line

Most of an MoE's weights (the experts) aren't used for any given token — so most
of the weights don't need to be in RAM. Keep them on disk, stream only the top-k
the router picks. See [`engine/README.md`](engine/README.md) for the mechanism and
numbers.
