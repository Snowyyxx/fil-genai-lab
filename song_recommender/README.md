# Agentic Song Recommender (LangGraph + web search)

An LLM **agent** that recommends songs by **searching the live web** — no local
catalog. Built on the local `llm_gateway` stack.

```
Browser (card UI) ─► FastAPI (:8100) ─► LangGraph state machine
                                          agent → tools → force_submit → fallback  (guardrail inline)
                                          tools: search_songs (iTunes API 🌐)
                                                 submit_recommendations
                          each LLM call ─► LiteLLM (:4000) ─► Ollama (qwen2.5:3b)
                          graph auto-traced ─► LangSmith (project: song-recommender)
                          likes ─► likes.json (no database)
```

**Orchestration:** LangGraph (`agent.py`) — control flow is a state machine
(nodes + conditional edges) rather than a hand-rolled loop.

- **Web-grounded, catalog-free:** the agent calls `search_songs`, which queries
  the **iTunes Search API** (keyless, millions of tracks) and returns STRUCTURED
  songs `{title, artist, genre, year, url}`. Any language/genre — not a fixed list.
- **Grounded picks:** submissions are validated against the returned candidates,
  so the model cannot invent tracks.
- **Cosine guardrail:** a pick is dropped if its title is embedding-similar to a
  song the user already gave (seed or liked) — catches remixes / '(From ...)'
  variants without an extra LLM call (`_drop_excluded`, threshold `RECO_GUARDRAIL_SIM`).
- **Agentic:** the model plans → `search_songs` → `submit_recommendations`.
- **Learns from feedback:** 👍/👎 writes `likes.json`; liked songs are injected
  into the next request's prompt (no DB).
- **Traced:** every agent run + tool + LLM call nests into one LangSmith trace.
- **Local model:** inference via Ollama through the LiteLLM gateway.

## Run

```bash
# gateway must be up:  cd ../llm_gateway && docker compose up -d   (qwen2.5:3b pulled)
python -m venv .venv-reco
.venv-reco/bin/pip install -r song_recommender/requirements.txt
.venv-reco/bin/python -m uvicorn song_recommender.main:app --host 0.0.0.0 --port 8100
```

Open **http://localhost:8100/**. A recommendation takes ~90s (3B model on CPU;
a GPU or a bigger-RAM 7B makes it fast).

## API

| Method | Path         | Body                                              |
|--------|--------------|---------------------------------------------------|
| POST   | `/recommend` | `{user_id, language, genre, mood, seed, count}`   |
| POST   | `/feedback`  | `{user_id, title, artist, liked}`                 |
| GET    | `/profile`   | `?user_id=demo`                                   |

## Notes

- **Retrieval = a structured music API.** We first tried generic DuckDuckGo text
  search, but `qwen2.5:3b` couldn't reliably *extract* song names from prose (it
  echoed page titles like "Qawwali - Wikipedia"). Switching to the **iTunes Search
  API** turned the job into *picking* structured rows, which small models do well:
  418s+garbage → ~88s and clean picks.
- **`mode`:** `agent` = LLM completed the loop; `fallback` = raw web results if
  the model misfired.
- **Postgres is no longer used** by the recommender (it still backs the LiteLLM
  console). The song catalog, embeddings, and DB tables were removed.
