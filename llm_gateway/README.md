# Open-Source LLM Gateway — FastAPI → LiteLLM → Ollama

A minimal, fully local stack for serving and calling **open-source** LLMs.

```
 FastAPI (:8000)  ──►  LiteLLM proxy (:4000)  ──►  Ollama (:11434)
   your app            OpenAI-compatible            open-source models
   (app/main.py)       gateway / router             (llama3.2, qwen2.5)
```

- **Ollama** actually runs the models (llama3.2:1b, qwen2.5:1.5b by default).
- **LiteLLM** puts a single OpenAI-compatible API in front of them, so your code
  never changes even if you later swap the backend.
- **FastAPI** is a tiny app that calls the gateway with the standard `openai`
  client — proving the whole thing works end to end.

Everything is open-source and runs on CPU. No API keys to anyone.

## Quick start

```bash
cd llm_gateway
cp .env.example .env          # optional: change the shared key

docker compose up -d          # first run pulls ~2 GB of models (one time)
docker compose logs -f ollama-pull   # watch the model download; exits when done
```

Once `ollama-pull` has finished and `litellm` is healthy, open the **chat UI**:

> 💬 **<http://localhost:8000/>** — a single-page chat: model dropdown, optional
> system prompt, temperature slider, streaming replies, multi-turn memory.
> Served by the FastAPI app itself (`app/static/index.html`); no build step.

Or drive it from the terminal:

```bash
# 1. Is the gateway alive and what does it serve?
curl localhost:8000/health

# 2. Ask a question through the FastAPI app
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain RAG in one sentence.", "model": "llama3.2"}'

# 3. Stream the tokens
curl -N -X POST localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a haiku about databases.", "model": "qwen2.5"}'
```

Interactive API docs: <http://localhost:8000/docs>

You can also hit LiteLLM directly (it's OpenAI-compatible), which is handy for
verifying the middle layer:

```bash
curl localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role":"user","content":"hi"}]}'
```

## The FastAPI endpoints

| Method | Path           | Body                                   | Returns                    |
|--------|----------------|----------------------------------------|----------------------------|
| GET    | `/`            | –                                      | the chat web UI            |
| GET    | `/health`      | –                                      | status + served models     |
| GET    | `/models`      | –                                      | model names                |
| POST   | `/chat`        | `{message, model?, system?, history?, temperature?, max_tokens?}` | `{model, reply, tokens}` |
| POST   | `/chat/stream` | same                                   | streamed text              |

`history` is an optional list of prior `{role, content}` turns — the chat UI
sends it automatically to give the model conversation memory.

## Adding / changing models

1. Pull it into Ollama:
   ```bash
   docker compose exec ollama ollama pull mistral:7b
   ```
2. Register a friendly name in [`litellm/config.yaml`](litellm/config.yaml):
   ```yaml
   - model_name: mistral
     litellm_params:
       model: ollama_chat/mistral:7b
       api_base: http://ollama:11434
   ```
3. Restart the proxy: `docker compose restart litellm`. Now call `"model": "mistral"`.

Browse available models at <https://ollama.com/library>. Smaller tags (`:1b`,
`:0.5b`, `:1.5b`) run comfortably on CPU; 7B+ models want a GPU (uncomment the
`deploy:` block for `ollama` in `docker-compose.yml`).

## Running the FastAPI app outside Docker (dev loop)

Keep `ollama` + `litellm` in Docker, run the app on your host with hot-reload:

```bash
cd app
pip install -r requirements.txt
export LITELLM_BASE_URL=http://localhost:4000/v1
export LITELLM_API_KEY=sk-local-dev-key
uvicorn main:app --reload --port 8000
```

## Why this shape?

- **LiteLLM in the middle** means your application code is provider-agnostic. To
  move from open-source local models to a hosted provider later, you change
  `litellm/config.yaml` — not your app.
- **The `openai` client** is all you need on the app side because LiteLLM speaks
  the OpenAI wire format.

## Troubleshooting

- **`/health` returns 503** → LiteLLM or Ollama isn't up yet. `docker compose ps`
  and `docker compose logs litellm`.
- **First `/chat` is slow / times out** → the model is loading into memory on
  first use, and CPU inference is slow. Retry; subsequent calls are faster. The
  proxy timeout is set to 600s in the config.
- **`model not found`** → the Ollama pull hasn't finished, or the name in your
  request doesn't match a `model_name` in `litellm/config.yaml`.

## Stop / clean up

```bash
docker compose down          # stop containers
docker compose down -v       # also delete the downloaded models (ollama_data volume)
```
