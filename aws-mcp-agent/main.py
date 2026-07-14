"""
Multi-Solution Math Agent — now a REAL MCP tool-calling agent.

WHAT CHANGED vs. the old version:
  Before: main.py imported MultiServerMCPClient but never used it. The "agent"
          was just an LLM in a loop; the MCP calculator tool was dead code.
  Now:    at startup we launch the MCP server (math_tools.py) over stdio, load
          its tools, and hand them to a LangGraph ReAct agent. The LLM can now
          actually CALL calculate_expression to compute/verify — true tool use.

ARCHITECTURE (one container):
   HTTP client ─► FastAPI /solve ─► LangGraph ReAct agent ─► Bedrock (Claude)
                                            │  (decides to call a tool)
                                            ▼
                                   MCP tool: calculate_expression
                                   (subprocess: python math_tools.py, stdio)

CREDENTIALS: boto3/langchain-aws pick these up from the environment.
  - Locally: from the .env you exported / python-dotenv loads.
  - On ECS Fargate: from the TASK ROLE automatically — no keys in the image.
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()  # local convenience; on ECS the env comes from the task role

# Amazon Nova needs no "use case form" (unlike Anthropic models on Bedrock) and
# supports tool-calling via the Converse API. Note the regional inference-profile
# prefix (eu.*) — the bare id fails with ValidationException in this region.
# To use Claude instead, submit the Anthropic use-case form in the Bedrock console
# and set BEDROCK_MODEL_ID=eu.anthropic.claude-haiku-4-5-20251001-v1:0
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.amazon.nova-lite-v1:0")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

# Holds the compiled agent once tools are loaded (see lifespan below).
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: launch the MCP server, load its tools, build agent."""
    client = MultiServerMCPClient(
        {
            "math": {
                # sys.executable => same interpreter (venv locally, python in container)
                "command": sys.executable,
                "args": ["math_tools.py"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()  # discovers calculate_expression over MCP

    llm = ChatBedrockConverse(model=MODEL_ID, region_name=REGION)

    # create_react_agent = LLM + tools in a reason/act loop (LangGraph prebuilt).
    STATE["agent"] = create_react_agent(llm, tools)
    STATE["tools"] = [t.name for t in tools]
    print(f"✓ agent ready | model={MODEL_ID} | tools={STATE['tools']}")
    yield
    STATE.clear()


app = FastAPI(title="Multi-Solution Math Agent (MCP)", lifespan=lifespan)

# CORS: harmless here (the UI is same-origin) but lets you call the API from
# other tools/origins during development.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):
    problem: str
    target_solutions: int = 2


class ChatRequest(BaseModel):
    message: str


def _text(content) -> str:
    """Bedrock/Nova returns message content as either a string or a list of
    typed blocks (text, reasoning_content, ...). Flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts).strip() or str(content)
    return str(content)


@app.get("/health")
async def health():
    """Liveness probe for ECS/ALB. Reports whether the agent + tools are loaded."""
    return {"status": "ok", "model": MODEL_ID, "tools": STATE.get("tools", [])}


@app.post("/solve")
async def solve_math_problem(request: QueryRequest):
    agent = STATE.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")

    system = (
        f"You are a math solver. Provide {request.target_solutions} fundamentally "
        f"DISTINCT approaches to the problem. Use the calculate_expression tool to "
        f"verify any arithmetic. Number each approach clearly."
    )
    try:
        result = await agent.ainvoke(
            {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request.problem},
            ]}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")

    # The final assistant message holds the answer; earlier messages include the
    # tool calls the agent made along the way.
    final = _text(result["messages"][-1].content)
    tool_calls = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
    return {
        "problem": request.problem,
        "answer": final,
        "tool_invocations": len(tool_calls),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """Freeform chat with the agent. It can use the calculate_expression tool
    when the question needs arithmetic — the frontend talks to this."""
    agent = STATE.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")
    system = (
        "You are a helpful assistant. When a question involves arithmetic, use the "
        "calculate_expression tool to compute the exact value rather than guessing."
    )
    try:
        result = await agent.ainvoke(
            {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request.message},
            ]}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")
    tool_calls = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
    return {"answer": _text(result["messages"][-1].content),
            "tool_invocations": len(tool_calls)}


@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Serve the chat UI. Same-origin as /chat, so no CORS/mixed-content issues."""
    return FRONTEND_HTML


FRONTEND_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MCP Agent · Bedrock + LangGraph</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0b1020; color:#e6e9f0; display:flex; flex-direction:column; height:100vh; }
  header { padding:14px 18px; border-bottom:1px solid #1e2740; background:#0e1428;
           display:flex; align-items:center; gap:10px; }
  header .dot { width:9px; height:9px; border-radius:50%; background:#38d39f; box-shadow:0 0 8px #38d39f; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .sub { font-size:12px; color:#8b93a7; margin-left:auto; }
  #log { flex:1; overflow-y:auto; padding:20px; display:flex; flex-direction:column; gap:14px; }
  .msg { max-width:760px; padding:12px 15px; border-radius:12px; white-space:pre-wrap;
         line-height:1.5; font-size:14px; }
  .user { align-self:flex-end; background:#2447d6; color:#fff; border-bottom-right-radius:3px; }
  .bot  { align-self:flex-start; background:#151d33; border:1px solid #223055; border-bottom-left-radius:3px; }
  .meta { font-size:11px; color:#7b84a0; margin-top:6px; }
  .badge { display:inline-block; background:#10243a; color:#8fd0ff; border:1px solid #294a70;
           border-radius:20px; padding:1px 8px; font-size:11px; }
  form { display:flex; gap:10px; padding:14px 18px; border-top:1px solid #1e2740; background:#0e1428; }
  input { flex:1; background:#0b1020; border:1px solid #26314f; color:#e6e9f0; border-radius:10px;
          padding:12px 14px; font-size:14px; outline:none; }
  input:focus { border-color:#3b62ff; }
  button { background:#3b62ff; color:#fff; border:0; border-radius:10px; padding:0 18px;
           font-size:14px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .hint { align-self:center; color:#6b7391; font-size:12px; }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>MCP Agent</h1>
    <span class="sub" id="sub">LangGraph · Bedrock Nova · MCP tool · ECS Fargate</span>
  </header>
  <div id="log">
    <div class="hint">Ask anything. Arithmetic is computed via the MCP <b>calculate_expression</b> tool.
    Try: “What is 47*89 + sqrt(144)?”</div>
  </div>
  <form id="f">
    <input id="q" placeholder="Type a message…" autocomplete="off" autofocus />
    <button id="send" type="submit">Send</button>
  </form>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('f');
  const input = document.getElementById('q');
  const send = document.getElementById('send');

  fetch('/health').then(r=>r.json()).then(d=>{
    document.getElementById('sub').textContent =
      (d.model||'') + ' · tools: ' + (d.tools||[]).join(', ');
  }).catch(()=>{});

  function add(cls, text){
    const el = document.createElement('div');
    el.className = 'msg ' + cls; el.textContent = text;
    log.appendChild(el); log.scrollTop = log.scrollHeight; return el;
  }

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const q = input.value.trim(); if(!q) return;
    add('user', q); input.value=''; send.disabled=true;
    const thinking = add('bot', '…');
    try{
      const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
                                      body: JSON.stringify({message:q})});
      const d = await r.json();
      if(!r.ok){ thinking.textContent = 'Error: ' + (d.detail||r.status); }
      else {
        thinking.textContent = d.answer;
        const m = document.createElement('div'); m.className='meta';
        m.innerHTML = '<span class="badge">🔧 tool calls: '+(d.tool_invocations||0)+'</span>';
        thinking.appendChild(m);
      }
    }catch(err){ thinking.textContent = 'Network error: ' + err; }
    finally { send.disabled=false; input.focus(); log.scrollTop = log.scrollHeight; }
  });
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
