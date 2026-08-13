# Building the "run a huge MoE model on tiny RAM" trick — from scratch

Goal: understand and rebuild the technique from the kimi-k3-in-c article — a model
that lives on **disk** while only the weights each token needs get pulled into
**RAM**, exploiting Mixture-of-Experts (MoE) sparsity.

We learn the mechanism in **Python first** (you can see every number), then port
the hot path to **C** at the end. Python's `np.memmap` uses the same OS lazy-paging
trick as C's `mmap`, so the concept transfers directly.

## The one idea, in one sentence
MoE layers only use ~k of N experts per token, so if the experts sit **contiguously
on disk** and you **memory-map** the file, touching k experts pages in only k
experts' worth of bytes — RAM stays tiny no matter how big the model is.

## Phases

1. **Streaming MoE layer (isolated)** — `phase1_moe_layer.py`
   One MoE block. A big file of experts on disk; read only the selected ones.
   Measure RAM: streaming stays flat, "load everything" balloons. ← the core trick.

2. **A full tiny transformer** — attention + RMSNorm + FFN forward pass on toy
   weights. Learn what an LLM forward pass actually is.

3. **Combine** — a transformer whose FFN layers are streaming-MoE blocks.

4. **Real small MoE model** — load a real tiny MoE (e.g. OLMoE / Qwen-MoE) via
   memory-mapped safetensors; add a real tokenizer and quantized weights.

5. **Port the hot path to C** — the "176 KB, no framework" version: `mmap`,
   pointers, on-the-fly dequant.

## Prereqs we'll pick up as we go
- Transformers: attention, RMSNorm, SwiGLU/ReLU FFN, and **MoE routing**.
- Systems: `mmap` / lazy paging, why disk layout matters.
- Quantization: packing weights into few bits and dequantizing.

Run phase 1:  `python moe_streaming/phase1_moe_layer.py`
