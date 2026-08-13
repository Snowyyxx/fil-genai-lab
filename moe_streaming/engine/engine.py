"""
StreamingMoE — a REAL Granite MoE whose experts live on disk and are streamed in
by OUR code, one top-k slice at a time, per token.

What transformers still does (so answers stay correct): tokenizer, attention,
norms, and the *router* that decides which experts to use.

What WE do (the whole point of the project):
  1. rip every expert's weights out of RAM into a flat file (experts.bin),
  2. patch each MoE block so that, per token, we read ONLY the experts the router
     picked — off disk, via mmap — and (in streaming mode) drop them again so the
     next token must re-read from disk.

Run standalone to sanity-check:
    python moe_streaming/engine/engine.py "your prompt"
"""
from __future__ import annotations

import gc
import json
import os
import types

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from disk_bank import DiskExpertBank

MODEL = "ibm-granite/granite-3.0-1b-a400m-instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "experts.bin")            # all expert weights, flat, on disk
MANIFEST = os.path.join(HERE, "experts.manifest.json")   # where each expert lives in BIN


def _streaming_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Drop-in replacement for GraniteMoeExperts.forward.

    It is a VERBATIM copy of the transformers implementation with exactly two
    lines changed (marked below): instead of indexing the in-RAM parameters
    self.gate_up_proj[i] / self.down_proj[i], we read those two matrices FROM DISK
    via self._bank. Everything else — which experts run, the SwiGLU math, the
    weighted scatter-add back — is identical, so the output is bit-for-bit the same
    as the stock model. Only *where the weights come from* changes.

    `top_k_index`   : [tokens, 8]  which experts each token routed to (from router)
    `top_k_weights` : [tokens, 8]  softmax gate for each of those picks
    """
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        # expert_mask[e] tells us which (token, slot) pairs picked expert e;
        # expert_hit is the list of experts that got at least one token this pass.
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    fired = []                                  # experts actually used (for the live UI)
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:      # defensive guard kept from the original
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]   # only the tokens routed to this expert
        i = int(expert_idx)
        # === the whole point: these two weight matrices come off DISK, not RAM ===
        gu = self._bank.read_expert(self._gu, i)   # gate+up proj  [2*inter, hidden]
        dn = self._bank.read_expert(self._dn, i)   # down proj     [hidden, inter]
        # ======================================================================
        gate, up = nn.functional.linear(current_state, gu).chunk(2, dim=-1)  # SwiGLU
        current = self.act_fn(gate) * up
        current = nn.functional.linear(current, dn)
        current = current * top_k_weights[token_idx, top_k_pos, None]        # apply gate
        final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        fired.append(i)

    self._bank.last_experts = fired
    return final_hidden_states


class StreamingMoE:
    def __init__(self, stream=True, verbose=True):
        self.log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
        # float32 + CPU: numpy-friendly and correct on any machine (no GPU needed).
        self.log(f"loading {MODEL} (float32, CPU)...")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
        self.num_experts = int(self.model.config.num_local_experts)      # 32
        self.active = int(self.model.config.num_experts_per_tok)         # 8 run per token
        self.layers = int(self.model.config.num_hidden_layers)           # 24

        # one-time: dump every expert to disk so we can drop it from RAM
        if not (os.path.exists(BIN) and os.path.exists(MANIFEST)):
            self._build_experts_bin()

        # open the on-disk expert file and tell the bank where each expert lives
        manifest = json.load(open(MANIFEST))
        self.bank = DiskExpertBank(BIN, dtype=np.float32)
        for m in manifest["modules"]:
            self.bank.register(m["name"], m["offset"], m["num_experts"], m["out"], m["in"])

        self._patch_and_free()           # swap experts for disk reads + free their RAM
        self.bank.stream = stream        # True = force disk reads; False = allow caching
        self.bank.cold()                 # evict experts.bin from page cache -> reads hit disk
        gc.collect()
        # token ids that end generation (eos, and the granite end marker if present)
        self.eos_ids = set(
            i for i in [self.tok.eos_token_id,
                        getattr(self.tok, "convert_tokens_to_ids", lambda x: None)("<|end_of_text|>")]
            if isinstance(i, int) and i >= 0)
        self.log(f"ready. experts on disk = {self.bank.stats()['total_experts_bytes']/1e9:.2f} GB, "
                 f"streaming={'ON' if stream else 'OFF'}")

    # ---- build / patch ----
    def _build_experts_bin(self):
        """One-time: write every expert weight tensor (the 3-D params, shape
        [num_experts, out, in]) into one flat file, back to back, and record where
        each module's block starts. Expert #i is then a single contiguous slice."""
        self.log(f"building {BIN} (one-time)...")
        manifest = {"model": MODEL, "dtype": "float32", "modules": []}
        offset = 0
        with open(BIN, "wb") as f:
            for name, p in self.model.named_parameters():
                if p.dim() != 3:                 # only the parallel-expert tensors are 3-D
                    continue
                num_experts, out, in_ = p.shape
                arr = p.detach().to(torch.float32).contiguous().numpy()
                f.write(arr.tobytes())
                manifest["modules"].append({"name": name, "offset": offset,
                                            "num_experts": int(num_experts),
                                            "out": int(out), "in": int(in_)})
                offset += arr.nbytes
        json.dump(manifest, open(MANIFEST, "w"), indent=2)
        self.log(f"  wrote {offset/1e9:.2f} GB across {len(manifest['modules'])} modules")

    def _patch_and_free(self):
        """For each MoE block: redirect its forward to read experts from disk, then
        DELETE the in-RAM expert parameters. Deleting from ._parameters drops the
        last reference to those big tensors, so Python frees the ~4.8 GB — that RAM
        drop is the whole payoff."""
        n = 0
        for name, mod in self.model.named_modules():
            if type(mod).__name__ != "GraniteMoeExperts":
                continue
            base = name  # e.g. model.layers.0.block_sparse_moe.experts
            mod._bank = self.bank
            mod._gu = f"{base}.gate_up_proj"     # manifest keys for this block's experts
            mod._dn = f"{base}.down_proj"
            mod.forward = types.MethodType(_streaming_experts_forward, mod)  # our forward
            for pname in ("gate_up_proj", "down_proj"):
                if pname in mod._parameters:
                    del mod._parameters[pname]   # free the RAM; now served from disk
            n += 1
        self.log(f"  patched {n} MoE blocks -> experts streamed from disk")

    # ---- generation ----
    @torch.no_grad()
    def generate_stream(self, messages, max_new_tokens=160):
        """Greedy decode, yielding (token_text, per-token stats) as we go.
        Standard two-phase loop: one big 'prefill' pass over the prompt, then one
        cheap pass per new token using the KV cache. Each pass triggers our disk
        reads inside the patched MoE blocks."""
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        ids = enc["input_ids"]
        self.bank.reset_counters()
        out = self.model(input_ids=ids, use_cache=True)          # prefill (whole prompt)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)         # greedy: most likely token

        for _ in range(max_new_tokens):
            tid = int(nxt)
            if tid in self.eos_ids:
                break
            stats = self.bank.stats()               # disk reads for the pass that made this token
            piece = self.tok.decode([tid], skip_special_tokens=True)
            yield piece, stats
            self.bank.reset_counters()
            out = self.model(input_ids=nxt, past_key_values=past, use_cache=True)  # 1-token step
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)


if __name__ == "__main__":
    import sys, time
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "In one sentence, what is a mixture-of-experts model?"
    eng = StreamingMoE(stream=True)
    print(f"\n>>> {prompt}\n")
    t0 = time.time(); ntok = 0; last = None
    for piece, stats in eng.generate_stream([{"role": "user", "content": prompt}], max_new_tokens=80):
        print(piece, end="", flush=True); ntok += 1; last = stats
    dt = time.time() - t0
    print(f"\n\n--- {ntok} tokens in {dt:.1f}s = {ntok/dt:.2f} tok/s ---")
    print(f"last-token: experts fired (final layer)={last['last_experts']}  "
          f"disk read={last['bytes_read']/1e6:.0f} MB  "
          f"major faults={last['majflt']}  streaming={last['streaming']}")
