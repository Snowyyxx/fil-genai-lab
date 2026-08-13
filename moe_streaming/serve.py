"""
A tiny, dependency-free server for the MoE chat interface.

Standard library only — no framework — matching the project's "from scratch"
spirit. It:
  • serves the web/ UIs (chat.html at /, plus demo.html and guide.html)
  • runs OUR streaming engine (engine/) — real MoE, experts read off disk — OR
    proxies to a local Ollama model, and STREAMS tokens back either way
  • exposes /stats — LIVE memory + disk-streaming counters (RAM-resident vs
    on-disk, and the major page faults our engine causes reading experts).

Layout:  serve.py · engine/ · web/ · examples/ · docs/
Run:      python moe_streaming/serve.py   →   http://localhost:8200
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("MOE_MODEL", "granite3-moe:3b")
PORT = int(os.environ.get("MOE_PORT", "8200"))
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")          # browser UIs (chat/demo/guide)
ENGINE_DIR = os.path.join(HERE, "engine")   # our from-scratch streaming engine

# ---- OUR from-scratch streaming engine (lazy-loaded) --------------------------
# Sentinel model name the UI sends to use our own MoE-from-disk engine instead of
# the Ollama black box.
OURS = "granite-moe (ours · streams from disk)"
_ENG = None
_ENG_LOCK = threading.Lock()   # only one generation at a time (model isn't reentrant)
_ENG_ERR = None


def get_engine():
    """Load StreamingMoE once, on first use. Returns (engine, error_str)."""
    global _ENG, _ENG_ERR
    if _ENG is not None or _ENG_ERR is not None:
        return _ENG, _ENG_ERR
    with _ENG_LOCK:
        if _ENG is None and _ENG_ERR is None:
            try:
                sys.path.insert(0, ENGINE_DIR)     # so `import engine` finds engine/engine.py
                from engine import StreamingMoE
                _ENG = StreamingMoE(stream=True, verbose=True)
            except Exception as e:
                _ENG_ERR = f"{type(e).__name__}: {e}"
    return _ENG, _ENG_ERR


def _self_stats():
    """Our OWN process: resident RAM + cumulative major faults (disk page-ins)."""
    rss = maj = 0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
    except Exception:
        pass
    try:
        c = open("/proc/self/stat").read()
        maj = int(c[c.rfind(")") + 2:].split()[9])
    except Exception:
        pass
    return rss, maj


def _ollama(path, body=None, timeout=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=timeout)


def _meminfo():
    """System memory from /proc/meminfo, in kB."""
    m = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            m[k.strip()] = int(v.strip().split()[0])
    except Exception:
        pass
    return m


def _find_runner():
    """The Ollama model-server subprocess (the one holding the weights) =
    the matching process with the largest resident memory."""
    best = None
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cl = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if ("runner" in cl or "llama_server" in cl) and ("ollama" in cl or "model" in cl):
            rss = _rss_kb(pid)
            if best is None or rss > best[1]:
                best = (pid, rss)
    return best  # (pid, rss_kb) or None


def _rss_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def _vsz_kb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmSize:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def _faults(pid):
    """(major, minor) page-fault counters from /proc/PID/stat.
    A MAJOR fault = a page had to be read from disk — i.e. streaming.
    (comm is in parens and may contain spaces, so parse after the last ')'.)"""
    try:
        c = open(f"/proc/{pid}/stat").read()
        rest = c[c.rfind(")") + 2:].split()
        return int(rest[9]), int(rest[7])   # majflt (field 12), minflt (field 10)
    except Exception:
        return 0, 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)
        if p.path in ("/", "/index.html", "/chat.html"):
            self._serve_web("chat.html")
        elif p.path in ("/demo.html", "/guide.html"):   # sibling pages linked from chat
            self._serve_web(p.path.lstrip("/"))
        elif p.path == "/health":
            self._json(self._model_info(q.get("model", [MODEL])[0]))
        elif p.path == "/models":
            self._json({"models": self._models(), "default": MODEL})
        elif p.path == "/stats":
            self._json(self._stats(q.get("model", [MODEL])[0]))
        else:
            self._head(404, "text/plain"); self.wfile.write(b"not found")

    def do_POST(self):
        if self.path != "/chat":
            self._head(404, "text/plain"); self.wfile.write(b"not found"); return
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model") or MODEL
        messages = req.get("messages") or [{"role": "user", "content": req.get("prompt", "")}]

        # --- OUR engine path: real MoE, experts streamed from disk by our code ---
        if model == OURS or req.get("engine") == "ours":
            self._chat_ours(messages, req)
            return

        # --- default path: proxy to Ollama ---
        self._head(200, "text/plain; charset=utf-8")
        try:
            with _ollama("/api/chat", {"model": model, "messages": messages, "stream": True}) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    tok = obj.get("message", {}).get("content", "")
                    if tok:
                        self.wfile.write(tok.encode("utf-8")); self.wfile.flush()
                    if obj.get("done"):
                        break
        except Exception as e:
            try:
                self.wfile.write(f"\n[error talking to Ollama: {e}]".encode())
            except Exception:
                pass

    def _chat_ours(self, messages, req):
        """Stream a reply from OUR engine: real Granite MoE, experts read off disk
        one top-k slice at a time by our own code."""
        eng, err = get_engine()
        if err or eng is None:
            self._head(200, "text/plain; charset=utf-8")
            self.wfile.write(f"[engine failed to load: {err}]".encode()); return
        # per-request streaming toggle: True = force disk reads (slow, real),
        # False = let the OS keep experts cached (fast after warmup)
        eng.bank.stream = bool(req.get("stream_disk", True))
        if eng.bank.stream:
            eng.bank.cold()   # evict so this turn's reads honestly come from disk
        # (in cache mode we DON'T evict — experts stay warm in the page cache)
        max_new = int(req.get("max_new_tokens", 200))
        self._head(200, "text/plain; charset=utf-8")
        try:
            with _ENG_LOCK:   # serialize generations
                for piece, _stats in eng.generate_stream(messages, max_new_tokens=max_new):
                    self.wfile.write(piece.encode("utf-8")); self.wfile.flush()
        except Exception as e:
            try:
                self.wfile.write(f"\n[engine error: {e}]".encode())
            except Exception:
                pass

    # ---- helpers ----
    def _serve_web(self, name):
        """Serve a static page from web/ (only the known .html files)."""
        try:
            with open(os.path.join(WEB, name), "rb") as f:
                self._head(200, "text/html; charset=utf-8"); self.wfile.write(f.read())
        except FileNotFoundError:
            self._head(404, "text/plain"); self.wfile.write(b"not found")

    def _head(self, code, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers()

    def _json(self, obj):
        self._head(200, "application/json"); self.wfile.write(json.dumps(obj).encode())

    def _models(self):
        # our engine is always first — it's the point of the project
        out = [{"name": OURS, "size": 0, "ours": True}]
        try:
            with _ollama("/api/tags", timeout=5) as r:
                tags = json.loads(r.read())
            oll = [{"name": m.get("name"), "size": m.get("size", 0)} for m in tags.get("models", [])]
            oll.sort(key=lambda m: -m["size"])
            out += oll
        except Exception:
            pass
        return out

    def _disk_bytes(self, model):
        for m in self._models():
            if m["name"] == model:
                return m["size"]
        return 0

    def _model_info(self, model):
        if model == OURS:
            eng, err = (_ENG, _ENG_ERR)   # don't trigger a load just for /health
            if eng is not None:
                return {"ok": True, "model": OURS, "experts": eng.num_experts,
                        "active": eng.active, "layers": eng.layers, "arch": "granitemoe",
                        "ours": True, "loaded": True}
            return {"ok": True, "model": OURS, "experts": 32, "active": 8, "layers": 24,
                    "arch": "granitemoe", "ours": True, "loaded": False, "error": err}
        try:
            with _ollama("/api/show", {"model": model}, timeout=10) as resp:
                d = json.loads(resp.read())
            mi = d.get("model_info", {})
            g = lambda s: next((v for k, v in mi.items() if k.endswith(s)), None)
            return {"ok": True, "model": model, "experts": g("expert_count"),
                    "active": g("expert_used_count"), "layers": g("block_count"),
                    "arch": mi.get("general.architecture")}
        except Exception as e:
            return {"ok": False, "model": model, "error": str(e)}

    def _stats(self, model):
        mem = _meminfo()
        # --- our engine: report OUR process + live expert-streaming counters ---
        if model == OURS:
            rss, maj = _self_stats()
            bank = _ENG.bank.stats() if _ENG is not None else {}
            return {
                "model": OURS, "ours": True,
                "loaded": _ENG is not None,
                "disk_kb": bank.get("total_experts_bytes", 0) // 1024,   # experts.bin size
                "rss_kb": rss,                                           # OUR resident RAM
                "vsz_kb": 0,
                "mem_total_kb": mem.get("MemTotal", 0),
                "mem_avail_kb": mem.get("MemAvailable", 0),
                "majflt": maj,                        # cumulative disk page-ins (our process)
                "minflt": 0,
                "runner_pid": os.getpid(),
                "streaming": bank.get("streaming", True),        # disk-forced vs cached
                "last_experts": bank.get("last_experts", []),    # real routed experts
                "num_experts": _ENG.num_experts if _ENG else 32,
                "active": _ENG.active if _ENG else 8,
            }
        # --- default: the Ollama runner process ---
        runner = _find_runner()
        pid = runner[0] if runner else None
        maj, minf = _faults(pid) if pid else (0, 0)
        return {
            "model": model,
            "disk_kb": self._disk_bytes(model) // 1024,
            "rss_kb": _rss_kb(pid) if pid else 0,          # weights actually resident in RAM
            "vsz_kb": _vsz_kb(pid) if pid else 0,          # virtual (mmap) size
            "mem_total_kb": mem.get("MemTotal", 0),
            "mem_avail_kb": mem.get("MemAvailable", 0),
            "majflt": maj,                                 # cumulative major faults = disk page-ins
            "minflt": minf,
            "runner_pid": pid,
        }


if __name__ == "__main__":
    print(f"MoE chat → http://localhost:{PORT}   (default model: {MODEL}, via {OLLAMA})")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
