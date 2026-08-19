# 🐈 Meowstermind — a Zen-cat AI todo app

A cozy task list guarded by a peaceful Zen cat. It keeps deadlines and estimates,
runs a Pomodoro timer with a **Cat Nap** break mode, and asks an **open-weight
model on AWS Bedrock** (Llama · Mistral · DeepSeek · Qwen · gpt-oss) to break
overwhelming tasks into bite-sized steps and offer daily wisdom.

**Phase 1 (this document): everything runs locally.** Phase 2 containerises the
same code for AWS Fargate — see the sketch at the bottom.

```
to-do-list/
├── backend/           FastAPI + boto3 → Bedrock Converse API
│   ├── main.py            routes
│   ├── zen_cat.py         Bedrock client, prompts, offline fallback
│   ├── schemas.py         pydantic models
│   ├── storage.py         JSON-file task store
│   ├── config.py          env-driven settings
│   ├── .env.example
│   └── tests/             31 tests, no AWS needed
├── frontend/
│   └── index.html     self-contained UI (Tailwind CDN, no build step)
├── scripts/
│   └── setup_bedrock.sh   one-shot Bedrock setup via the aws CLI
└── README.md
```

### Why the Converse API, and why open weights

`bedrock-runtime.converse` is **one request shape for every Bedrock vendor**, so
switching models is a `BEDROCK_MODEL_ID` change and nothing else — no rewrite of
the request body. (Per-vendor `invoke_model` payloads would hard-code us to one
provider.) The app defaults to **open-weight** models because on Bedrock they
need only the standard model agreement, whereas Anthropic models additionally
require a vendor use-case form. Any model still works if you pin it explicitly.

---

## 1. Install

```bash
cd to-do-list
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt     # runtime + pytest
```

## 2. Run it (works immediately, no AWS needed)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** — FastAPI serves the frontend too, so that is the
whole setup. Interactive API docs live at **/docs**.

Without AWS credentials the cat answers from its offline brain and every AI
response is labelled `"source": "mock"` with a `note` explaining why. Nothing is
silently faked, and the entire UI is usable while you sort out access.

> Prefer a separate frontend server? `cd frontend && python3 -m http.server 5500`
> then open http://127.0.0.1:5500 — CORS is open and the page falls back to
> `http://127.0.0.1:8000` for the API. Override with
> `localStorage.setItem('meow_api', 'http://host:port')`.

## 3. Set up Bedrock with `aws` commands

### a. Install the AWS CLI (if you don't have it)

```bash
curl -sSfL -o /tmp/awscliv2.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
unzip -q /tmp/awscliv2.zip -d /tmp/awscli
/tmp/awscli/aws/install --install-dir ~/.local/aws-cli --bin-dir ~/.local/bin --update
export PATH="$HOME/.local/bin:$PATH"       # add to ~/.bashrc to make it stick
aws --version
```

### b. Authenticate — pick one

```bash
aws configure                    # IAM user access keys
aws configure sso                # IAM Identity Center, then: aws sso login --profile <name>
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...
```

### c. Run the setup script

```bash
./scripts/setup_bedrock.sh                      # read-only: what's available, what's granted
./scripts/setup_bedrock.sh --accept-terms       # also accept the model agreement
./scripts/setup_bedrock.sh --region us-east-1   # pick a region (widest choice)
./scripts/setup_bedrock.sh --iam-policy         # print the least-privilege IAM policy
```

It is **read-only unless you pass `--accept-terms`**, because creating a model
agreement is a legal acceptance on your account. In order it:

1. `aws sts get-caller-identity` — confirms credentials, prints account + region
2. `aws bedrock list-foundation-models` — open-weight models offered here
3. `aws bedrock list-inference-profiles` — several models are profile-only
4. `aws bedrock get-foundation-model-availability` — is access granted?
5. `aws bedrock list-foundation-model-agreement-offers` + `create-foundation-model-agreement` — accepts terms *(only with `--accept-terms`)*
6. `aws bedrock-runtime converse` — a real call, the same API the app uses

On success it prints the two exports to paste:

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.meta.llama3-3-70b-instruct-v1:0
```

### d. Confirm from the app

```bash
curl -s localhost:8000/health | python3 -m json.tool
# ai_mode: "bedrock", model_id: "...", detail: null  → live
curl -s localhost:8000/ai/models | python3 -m json.tool   # what this account can invoke
```

The header pill in the UI shows the same thing: 🧠 Bedrock vs 💤 offline cat.

**IAM permissions needed:** `bedrock:InvokeModel` (required), plus
`bedrock:ListFoundationModels` and `bedrock:ListInferenceProfiles` for
auto-discovery. `--iam-policy` prints a ready-made least-privilege policy.

**Model selection.** With `BEDROCK_MODEL_ID` unset, the app lists what your
account can invoke, keeps the open-weight vendors (meta, mistral, deepseek,
qwen, openai) and picks the most capable — preferring inference profiles, since
several models are only reachable that way. Pin the env var to override.

## 4. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | status, AI mode, resolved model id |
| `GET` | `/ai/models` | model ids invokable from this account |
| `GET` | `/tasks` | list tasks (unfinished first, then by deadline) |
| `POST` | `/tasks` | create `{title, deadline?, estimate_minutes?, notes?}` |
| `PATCH` | `/tasks/{id}` | update any field, e.g. `{"done": true}` |
| `DELETE` | `/tasks/{id}` | delete |
| `PATCH` | `/tasks/{id}/subtasks/{i}` | tick a subtask `{"done": true}` |
| `POST` | `/ai/breakdown` | split a task into subtasks + a zen comment |
| `POST` | `/ai/advice` | daily wisdom / what to do next |

```bash
# create a task
curl -s -X POST localhost:8000/tasks -H 'Content-Type: application/json' \
  -d '{"title":"Write the dissertation chapter","deadline":"2026-09-01","estimate_minutes":180}'

# break it down and save the steps onto that task
curl -s -X POST localhost:8000/ai/breakdown -H 'Content-Type: application/json' \
  -d '{"task_id":"<id>","max_subtasks":5}'

# preview a breakdown without touching the list
curl -s -X POST localhost:8000/ai/breakdown -H 'Content-Type: application/json' \
  -d '{"title":"Plan the move"}'

# wisdom — an empty body means "read my saved list"
curl -s -X POST localhost:8000/ai/advice -H 'Content-Type: application/json' -d '{}'
```

Tasks persist to `backend/tasks.json` (override with `MEOW_DATA_FILE`).

## 5. Test

```bash
cd backend
MEOW_FORCE_MOCK=1 pytest -q          # 31 tests, no AWS required
```

`tests/test_api.py` covers task CRUD, validation, subtask toggling, breakdown
persistence, advice prioritisation and 404s. `tests/test_bedrock.py` stubs the
boto3 client to cover the parts that would otherwise need an account: the
Converse request shape, the retry for models that reject a system block,
open-weight filtering, `<think>`-block stripping, JSON repair, and the fallback
when a model returns junk instead of JSON.

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `BEDROCK_MODEL_ID` | auto-discovered | pin a specific model |
| `BEDROCK_MAX_TOKENS` | `3000` | response cap |
| `MEOW_FORCE_MOCK` | `0` | `1` = never call Bedrock |
| `MEOW_DATA_FILE` | `backend/tasks.json` | task storage |
| `MEOW_ALLOW_ORIGINS` | `*` | CORS allowlist (tighten in prod) |

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `NoCredentialsError` in `/health.detail` | No credentials on the process. Export `AWS_PROFILE` **before** starting uvicorn. |
| `AccessDeniedException` | IAM lacks `bedrock:InvokeModel`, or the model agreement isn't accepted — run `./scripts/setup_bedrock.sh --accept-terms`. |
| `/ai/models` returns empty lists, no errors | Credentials fine, but no open-weight models enabled in that region. Try `--region us-east-1`. |
| `ValidationException: on-demand throughput isn't supported` | That model is profile-only — use the `us.*` / `eu.*` id from `/ai/models`. |
| Model returns prose instead of JSON | Handled: the reply is repaired, and if it can't be, the offline cat answers with a `note`. Smaller models do this more; try a 70B+ one. |
| `ThrottlingException` | Bedrock rate limit; retry, or pick a smaller model. |
| Page loads unstyled | Tailwind/fonts come from a CDN — needs internet. The API still works. |

---

## Phase 2 preview — containerise & ship to Fargate

The code is already Fargate-shaped: FastAPI serves the frontend from the same
process (one image, one port), all config is env-driven, and `/health` is a
ready-made ALB health check. When you're ready, Phase 2 adds:

- a slim `Dockerfile` (python:3.12-slim, non-root, `uvicorn --host 0.0.0.0 --port 8080`)
- `docker build` + push to **ECR**
- an ECS **Fargate** service behind an ALB, with a **task role** carrying the
  policy from `--iam-policy` (no keys in the image)
- swapping `storage.py`'s JSON file for DynamoDB, since Fargate tasks are ephemeral

Say the word and I'll build it.
