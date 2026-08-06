"""
FastAPI service for Faraz Shayari.

  GET  /         -> the chat UI (same-origin, so no CORS/mixed-content issues)
  POST /shayari  -> {situation} -> {shayari, inspirations}
  GET  /health   -> liveness probe for ECS

The heavy lifting (retrieve + generate) lives in generate.py; this file is a
thin HTTP shell. It talks only to the LiteLLM gateway (which reaches Bedrock).

The frontend HTML is served from S3 when HTML_URL is set (falls back to the
bundled frontend.html, then to an embedded copy) — so the web code can be
updated in the bucket without rebuilding the container.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from generate import make_shayari, GEN_MODEL

app = FastAPI(title="Faraz Shayari")

_HTML_CACHE: dict[str, str] = {}


def get_frontend() -> str:
    """Fetch the UI: from S3 (HTML_URL) if configured, else the local file,
    else the embedded fallback. S3 result is cached in memory after first fetch."""
    url = os.getenv("HTML_URL")
    if url:
        if "html" not in _HTML_CACHE:
            with urllib.request.urlopen(url, timeout=10) as r:
                _HTML_CACHE["html"] = r.read().decode("utf-8")
        return _HTML_CACHE["html"]
    local = Path(__file__).parent / "frontend.html"
    if local.exists():
        return local.read_text(encoding="utf-8")
    return FRONTEND_HTML


class Req(BaseModel):
    situation: str


@app.get("/health")
async def health():
    return {"status": "ok", "model": GEN_MODEL}


@app.post("/shayari")
async def shayari(req: Req):
    if not req.situation.strip():
        raise HTTPException(status_code=400, detail="situation is empty")
    try:
        return make_shayari(req.situation.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/", response_class=HTMLResponse)
async def home():
    return get_frontend()


FRONTEND_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Faraz Shayari</title>
<style>
  :root{--bg:#0e0b14;--card:#171320;--ink:#f2ecdf;--mute:#a99fb3;--accent:#d9a441;--line:#2a2436}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#251a2e,#0e0b14);
       color:var(--ink);font-family:Georgia,'Times New Roman',serif;min-height:100vh}
  .wrap{max-width:760px;margin:0 auto;padding:48px 20px 80px}
  h1{font-size:34px;letter-spacing:.02em;margin:0 0 4px;font-weight:600}
  h1 .sub{display:block;font-size:14px;color:var(--accent);letter-spacing:.28em;
          text-transform:uppercase;margin-bottom:10px;font-family:system-ui,sans-serif}
  p.tag{color:var(--mute);margin:0 0 28px;font-style:italic}
  .box{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px}
  textarea{width:100%;background:#0e0b14;border:1px solid var(--line);border-radius:10px;
           color:var(--ink);padding:14px;font-size:16px;font-family:inherit;resize:vertical;min-height:80px}
  button{margin-top:12px;background:var(--accent);color:#1a1206;border:0;border-radius:10px;
         padding:12px 22px;font-size:16px;font-weight:700;cursor:pointer;font-family:system-ui,sans-serif}
  button:disabled{opacity:.5;cursor:wait}
  .shayari{white-space:pre-wrap;font-size:22px;line-height:1.9;color:#fff;text-align:center;
           padding:26px 10px;letter-spacing:.01em}
  .meaning{color:var(--mute);font-style:italic;text-align:center;font-size:15px}
  .insp{margin-top:26px;border-top:1px solid var(--line);padding-top:16px}
  .insp h3{font-family:system-ui,sans-serif;font-size:12px;letter-spacing:.18em;text-transform:uppercase;
           color:var(--accent);margin:0 0 12px}
  .couplet{color:var(--mute);font-size:15px;line-height:1.7;margin:0 0 12px;padding-left:14px;
           border-left:2px solid var(--line)}
  .note{margin-top:22px;color:#6f6780;font-size:12px;font-family:system-ui,sans-serif;text-align:center}
  .ex{color:var(--accent);cursor:pointer;text-decoration:underline dotted}
</style></head><body>
<div class="wrap">
  <h1><span class="sub">In the voice of</span>Ahmad Faraz</h1>
  <p class="tag">Tell me what you are going through — I will answer in shayari.</p>
  <div class="box">
    <textarea id="s" placeholder="e.g. I still think about someone who has clearly forgotten me..."></textarea>
    <button id="go" onclick="ask()">Compose</button>
    <div style="margin-top:10px;font-size:13px;color:var(--mute)">
      try:
      <span class="ex" onclick="fill('I got the job I always wanted but no one to share it with')">a bittersweet win</span> ·
      <span class="ex" onclick="fill('I am waiting for someone who may never come back')">longing</span> ·
      <span class="ex" onclick="fill('I finally let go of the person I loved most')">letting go</span>
    </div>
  </div>
  <div id="out"></div>
  <div class="note">AI-composed in Ahmad Faraz's style — not his actual verse. The couplets below are his real work (source: Rekhta).</div>
</div>
<script>
function fill(t){document.getElementById('s').value=t;}
async function ask(){
  const s=document.getElementById('s').value.trim(); if(!s)return;
  const btn=document.getElementById('go'), out=document.getElementById('out');
  btn.disabled=true; btn.textContent='Composing…'; out.innerHTML='';
  try{
    const r=await fetch('/shayari',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({situation:s})});
    const d=await r.json();
    if(!r.ok){out.innerHTML='<div class="box" style="margin-top:20px;color:#e88">'+(d.detail||'error')+'</div>';return;}
    const parts=(d.shayari||'').split(/\\nmeaning:/i);
    const verse=parts[0].trim(), meaning=parts[1]?parts[1].trim():'';
    let html='<div class="box" style="margin-top:20px"><div class="shayari">'+verse+'</div>';
    if(meaning)html+='<div class="meaning">'+meaning+'</div>';
    if(d.inspirations&&d.inspirations.length){
      html+='<div class="insp"><h3>Inspired by Faraz\\'s real couplets</h3>';
      for(const c of d.inspirations){html+='<div class="couplet">'+c.couplet+'</div>';}
      html+='</div>';
    }
    html+='</div>'; out.innerHTML=html;
  }catch(e){out.innerHTML='<div class="box" style="margin-top:20px;color:#e88">'+e+'</div>';}
  finally{btn.disabled=false; btn.textContent='Compose';}
}
</script></body></html>"""
