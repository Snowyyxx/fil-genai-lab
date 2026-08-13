# The streaming engine — real MoE, experts on disk

This is the heart of the project: a **real, trained** Granite 3 MoE (1 B params,
32 experts, top-8, 24 layers) that runs with its experts **on disk**, streamed in
by our own code one top-k slice at a time — not by a black box.

```
prompt ─▶ tokenizer ─▶ [ per layer × 24 ]
                         attention (RAM)  ─▶  router picks 8 of 32 experts
                                                    │
                                          OUR CODE reads only those 8
                                          experts' weights off disk  ◀── experts.bin (4.83 GB)
                                                    │
                                          matmul ─▶ combine ─▶ next layer
                         ─▶ next token
```

**What transformers still does** (so answers are correct): tokenizer, attention,
norms, and the router that *chooses* experts.
**What we do** (the whole point): keep every expert's weights out of RAM, and per
token read only the experts the router asked for — off disk, via `mmap`.

## Result (measured, granite-3.0-1b-a400m)

| | |
|---|---|
| Experts on disk (`experts.bin`) | **4.83 GB** |
| Our process RAM (RSS) | **~1.0 GB** |
| → a ~5.8 GB model in | **< 1 GB RAM** |
| Speed, streaming from disk | ~0.9 tok/s (disk-bound — the lesson) |
| Speed, experts cached in RAM | faster (RAM-bound) |

## Files

- **`disk_bank.py`** — `DiskExpertBank`: memory-maps `experts.bin`, reads expert
  `#i` as a contiguous slice, and (in streaming mode) evicts it again with
  `madvise(DONTNEED)` + `posix_fadvise(DONTNEED)` so the next token must re-read
  it from the physical disk — a genuine **major page fault** we can count.
- **`engine.py`** — `StreamingMoE`: loads the real model, writes every expert to
  `experts.bin` (once), **deletes the in-RAM expert tensors**, and patches each
  `GraniteMoeExperts.forward` to read `gate_up_proj[e]` / `down_proj[e]` from the
  disk bank instead of RAM. Exposes `generate_stream()` with per-token stats.
- `experts.bin` + `experts.manifest.json` are **generated on first run** (git-ignored).

## Run it

```bash
# standalone (builds experts.bin on first run, ~one-time):
python moe_streaming/engine/engine.py "Explain mixture-of-experts briefly."

# or through the chat UI:
python moe_streaming/serve.py     # → http://localhost:8200  → pick "🧩 granite-moe (ours)"
```

## Why major faults, not just reads

If the file simply fits in the page cache, re-reading an expert is a *minor* fault
(RAM hit) and nothing streams. To make "streaming from disk" honest at any model
size, streaming mode **evicts** each expert after use: `madvise(MADV_DONTNEED)`
first (unmaps it from our process — `fadvise` won't drop still-mapped pages), then
`posix_fadvise(POSIX_FADV_DONTNEED)` (drops it from the OS page cache). The next
access is a real disk read. No root required.
