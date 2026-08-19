"""Meowstermind's brain: an open-weight model on AWS Bedrock, plus an offline cat.

Design notes
------------
* We call Bedrock through **boto3** `bedrock-runtime.converse`. The Converse API
  is one request/response shape for *every* Bedrock model, so swapping Llama for
  Mistral, DeepSeek or Qwen is a model-id change and nothing else. (Per-model
  `invoke_model` bodies are vendor-specific and would lock us in.)
* We deliberately prefer **open-weight** models — Llama, Mistral, DeepSeek, Qwen,
  gpt-oss. They need no vendor use-case form on Bedrock, only the standard model
  agreement. Set BEDROCK_MODEL_ID to override with anything you like.
* The model id is discovered from *your* account on first use, because which
  models are enabled differs per account/region and several are reachable only
  through a regional inference profile (`us.*` / `eu.*`).
* Every AI call has an offline fallback so the whole UI is testable without
  credentials. Responses carry `source: "bedrock" | "mock"` so nothing is ever
  silently faked.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading

from config import settings
from schemas import (
    AdviceRequest,
    AdviceResponse,
    BreakdownRequest,
    BreakdownResponse,
    Subtask,
)

log = logging.getLogger("meowstermind.cat")

# Providers whose weights are openly published. Discovery only considers these
# so we never silently fall back to a proprietary (or gated) model.
OPEN_WEIGHT_PROVIDERS = frozenset({"meta", "mistral", "deepseek", "qwen", "openai"})

# Most capable first. The first entry present in your account wins. These are
# substrings, not exact ids, so regional profiles (us.meta.llama…) match too.
MODEL_PREFERENCE = (
    "gpt-oss-120b",       # OpenAI gpt-oss, Apache-2.0 weights
    "llama4-maverick",
    "llama3-3-70b",
    "llama4-scout",
    "llama3-1-70b",
    "qwen3-235b",
    "mixtral-8x7b",
    "qwen3-32b",
    "deepseek",           # DeepSeek-R1 — reasoning model, emits <think> blocks
    "gpt-oss-20b",
    "mistral-small",
    "llama3-1-8b",
    "llama3-8b",
    "mistral-7b",
)

PERSONA = """You are Meowstermind: a peaceful, wise Zen cat who helps a human keep a cozy to-do list.

Voice:
- Warm, unhurried, gently encouraging. A calm friend, not a productivity drill sergeant.
- Short sentences. Concrete, never preachy. At most one soft cat-ism (a purr, a paw, a sunbeam) per reply.
- Never shame the human for an unfinished list. Overwhelm is normal; the next small step is the cure.

Rules:
- Reply with ONE JSON object and nothing else. Start at { and end at }.
- No markdown fences, no preamble, no explanation before or after the JSON.
- Keep every string plain text (no emoji spam, no bullet characters)."""


class CatBrain:
    """Holds the Bedrock clients, the resolved model id and the last error."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtime = None
        self._control = None
        self._model_id: str | None = settings.model_id
        self._resolved = settings.model_id is not None
        self.last_error: str | None = None

    # --- clients -----------------------------------------------------------
    def _clients(self):
        """Create the boto3 clients lazily so import never touches the network."""
        if self._runtime is None:
            import boto3  # imported here so `MEOW_FORCE_MOCK=1` needs no AWS deps at import

            self._runtime = boto3.client("bedrock-runtime", region_name=settings.aws_region)
            self._control = boto3.client("bedrock", region_name=settings.aws_region)
        return self._runtime, self._control

    # --- model discovery ---------------------------------------------------
    def available_models(self) -> dict:
        """List the open-weight ids this account can actually invoke, best first.

        Never raises: lookup failures (no credentials, denied IAM, region without
        Bedrock) come back in `errors` so the caller can report the real cause
        instead of guessing.
        """
        profiles: list[str] = []
        on_demand: list[str] = []
        errors: list[str] = []

        try:
            _, control = self._clients()
        except Exception as exc:
            return {
                "inference_profiles": [],
                "on_demand_models": [],
                "errors": [f"could not create Bedrock client: {exc}"],
            }

        try:
            paginator = control.get_paginator("list_inference_profiles")
            for page in paginator.paginate():
                for item in page.get("inferenceProfileSummaries", []):
                    pid = item.get("inferenceProfileId", "")
                    if _provider(pid) in OPEN_WEIGHT_PROVIDERS:
                        profiles.append(pid)
        except Exception as exc:  # ListInferenceProfiles may be denied or unsupported
            log.info("Could not list inference profiles: %s", exc)
            errors.append(f"list_inference_profiles: {_brief(exc)}")

        try:
            # No byProvider filter: one call, then keep the open-weight vendors.
            for item in control.list_foundation_models().get("modelSummaries", []):
                mid = item.get("modelId", "")
                if _provider(mid) not in OPEN_WEIGHT_PROVIDERS:
                    continue
                if "TEXT" not in item.get("outputModalities", ["TEXT"]):
                    continue
                if "ON_DEMAND" in item.get("inferenceTypesSupported", []):
                    on_demand.append(mid)
        except Exception as exc:
            log.info("Could not list foundation models: %s", exc)
            errors.append(f"list_foundation_models: {_brief(exc)}")

        return {
            "inference_profiles": _rank(profiles),
            "on_demand_models": _rank(on_demand),
            "errors": errors,
        }

    def resolve_model_id(self) -> str | None:
        """Return the model id to invoke, discovering one on first use."""
        with self._lock:
            if self._resolved:
                return self._model_id
            found = self.available_models()
            # Inference profiles first: several models are profile-only.
            candidates = found["inference_profiles"] + found["on_demand_models"]
            self._model_id = candidates[0] if candidates else None
            self._resolved = True
            if self._model_id:
                log.info("Meowstermind picked Bedrock model %s", self._model_id)
            elif found["errors"]:
                # Credentials, IAM or region problem - say which, don't guess.
                self.last_error = "could not reach Bedrock (" + "; ".join(found["errors"]) + ")"
            else:
                self.last_error = (
                    f"no open-weight models are enabled in {settings.aws_region} - "
                    "run scripts/setup_bedrock.sh --accept-terms, or set BEDROCK_MODEL_ID"
                )
            return self._model_id

    # --- invocation --------------------------------------------------------
    def invoke(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Send one message through the Converse API and return the text reply.

        Converse normalises every Bedrock vendor onto the same shape, so this
        works unchanged for Llama, Mistral, DeepSeek, Qwen, gpt-oss and others.
        """
        model_id = self.resolve_model_id()
        if not model_id:
            raise RuntimeError(self.last_error or "no Bedrock model available")

        runtime, _ = self._clients()
        request = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "system": [{"text": PERSONA}],
            "inferenceConfig": {
                "maxTokens": max_tokens or settings.max_tokens,
                "temperature": 0.7,
            },
        }

        try:
            response = runtime.converse(**request)
        except Exception as exc:
            # A few models reject a separate system block; fold it into the turn.
            if "system" not in str(exc).lower():
                raise
            log.info("Model %s rejected a system block; inlining the persona", model_id)
            request.pop("system")
            request["messages"] = [{"role": "user", "content": [{"text": f"{PERSONA}\n\n{prompt}"}]}]
            response = runtime.converse(**request)

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b["text"] for b in blocks if "text" in b).strip()
        if not text:
            raise RuntimeError(f"empty reply (stopReason={response.get('stopReason')})")
        self.last_error = None
        return text

    @property
    def mode(self) -> str:
        return "mock" if settings.force_mock else "bedrock"

    @property
    def model_id(self) -> str | None:
        return self._model_id


brain = CatBrain()


_REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.")


def _provider(model_id: str) -> str:
    """Vendor part of a Bedrock id, ignoring any regional-profile prefix.

    'us.meta.llama3-3-70b-instruct-v1:0' -> 'meta'
    'deepseek.r1-v1:0'                   -> 'deepseek'
    """
    parts = model_id.split(".")
    if len(parts) > 2 and f"{parts[0]}." in _REGION_PREFIXES:
        parts = parts[1:]
    return parts[0] if parts else ""


def _brief(exc: Exception) -> str:
    """One-line description of a boto3 failure, useful in an API response."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:180]}" if text else type(exc).__name__


def _rank(ids: list[str]) -> list[str]:
    """Sort model ids by our capability preference, dropping duplicates."""

    def key(model_id: str) -> tuple[int, str]:
        for index, name in enumerate(MODEL_PREFERENCE):
            if name in model_id:
                return (index, model_id)
        return (len(MODEL_PREFERENCE), model_id)

    return sorted(dict.fromkeys(ids), key=key)


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply, tolerating stray prose or fences.

    Open-weight models are chattier than Claude about output format, and
    reasoning models (DeepSeek-R1) prefix their answer with a <think> block,
    so this is deliberately forgiving.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("model reply contained no JSON object")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _breakdown_prompt(title: str, notes: str | None, max_subtasks: int) -> str:
    extra = f"\nExtra context from the human: {notes}" if notes else ""
    return f"""A human feels stuck on this task:

Task: {title}{extra}

Break it into at most {max_subtasks} bite-sized steps. Each step must be something
they could finish in one sitting, phrased as a concrete action starting with a verb.
Estimate honest minutes for each (5-90). Then add one short zen thought (max 25 words)
about starting.

Reply with exactly this JSON shape:
{{"subtasks": [{{"title": "...", "estimate_minutes": 20}}], "zen_comment": "..."}}"""


def _advice_prompt(req: AdviceRequest) -> str:
    if req.tasks:
        lines = []
        for t in req.tasks[:20]:
            bits = [f"- {t.title}"]
            if t.deadline:
                bits.append(f"(due {t.deadline})")
            if t.estimate_minutes:
                bits.append(f"[~{t.estimate_minutes} min]")
            if t.done:
                bits.append("[done]")
            lines.append(" ".join(bits))
        board = "Their list right now:\n" + "\n".join(lines)
    else:
        board = "Their list is empty."

    mood = f"\nThey say they feel: {req.mood}" if req.mood else ""
    return f"""{board}{mood}

Offer a moment of calm guidance. `wisdom` is 1-2 sentences of cozy encouragement
(max 40 words). `focus_suggestion` names the single task worth doing next and one
sentence of why - or null if the list is empty or finished.

Reply with exactly this JSON shape:
{{"wisdom": "...", "focus_suggestion": "..."}}"""


# --------------------------------------------------------------------------- #
# Offline cat (used when Bedrock is unreachable or MEOW_FORCE_MOCK=1)
# --------------------------------------------------------------------------- #
_MOCK_ZEN = (
    "A long path is only ever one paw-step wide. Take the first one.",
    "Even a mountain of yarn unravels from a single thread.",
    "The nap is earned by the beginning, not the finish. Begin gently.",
    "Do not carry the whole task. Carry only its first minute.",
)
_MOCK_WISDOM = (
    "The list will keep. Your attention will not. Give one task your whole soft focus.",
    "Small steps taken kindly outrun big plans made anxiously.",
    "Sit, breathe, pick one thing. The rest of the list can nap a while longer.",
    "A tidy mind beats a long list. Finish something small and feel the room lighten.",
)


def mock_breakdown(title: str, max_subtasks: int) -> BreakdownResponse:
    short = title.strip().rstrip(".")
    steps = [
        (f"Write one sentence describing what 'done' looks like for {short}", 5),
        (f"Gather everything you need for {short} in one place", 10),
        (f"Do the smallest visible piece of {short}", 25),
        (f"Work the messy middle of {short} for one focus block", 25),
        (f"Tidy up and note the next paw-step for {short}", 10),
        (f"Review {short} once with fresh eyes", 15),
        (f"Share or ship {short}", 15),
        (f"Close the loop on {short} and stretch", 5),
    ][:max_subtasks]
    return BreakdownResponse(
        subtasks=[Subtask(title=t, estimate_minutes=m) for t, m in steps],
        zen_comment=random.choice(_MOCK_ZEN),
        source="mock",
    )


def mock_advice(req: AdviceRequest) -> AdviceResponse:
    pending = [t for t in req.tasks if not t.done]
    if not req.tasks:
        return AdviceResponse(
            wisdom="An empty list is not laziness, it is space. Add one gentle intention.",
            focus_suggestion=None,
            source="mock",
        )
    if not pending:
        return AdviceResponse(
            wisdom="Everything is done. Close the laptop, find a sunbeam, and be pleased with yourself.",
            focus_suggestion=None,
            source="mock",
        )
    dated = [t for t in pending if t.deadline]
    pick = min(dated, key=lambda t: t.deadline or "") if dated else min(
        pending, key=lambda t: t.estimate_minutes or 9999
    )
    why = "it is nearest its deadline" if dated else "it is the lightest thing on the list"
    return AdviceResponse(
        wisdom=random.choice(_MOCK_WISDOM),
        focus_suggestion=f"Start with \"{pick.title}\" - {why}.",
        source="mock",
    )


# --------------------------------------------------------------------------- #
# Public API used by the routes
# --------------------------------------------------------------------------- #
def breakdown(req: BreakdownRequest, title: str) -> BreakdownResponse:
    if settings.force_mock:
        return mock_breakdown(title, req.max_subtasks)
    try:
        data = _extract_json(brain.invoke(_breakdown_prompt(title, req.notes, req.max_subtasks)))
        subtasks = [
            Subtask(
                title=str(item["title"]).strip(),
                estimate_minutes=int(item.get("estimate_minutes") or 15),
            )
            for item in data.get("subtasks", [])
            if str(item.get("title", "")).strip()
        ][: req.max_subtasks]
        if not subtasks:
            raise ValueError("model returned no usable subtasks")
        return BreakdownResponse(
            subtasks=subtasks,
            zen_comment=str(data.get("zen_comment") or random.choice(_MOCK_ZEN)).strip(),
            source="bedrock",
        )
    except Exception as exc:
        log.warning("Bedrock breakdown failed, using offline cat: %s", exc)
        brain.last_error = str(exc)
        fallback = mock_breakdown(title, req.max_subtasks)
        fallback.note = f"Offline cat answered — {exc}"
        return fallback


def advice(req: AdviceRequest) -> AdviceResponse:
    if settings.force_mock:
        return mock_advice(req)
    try:
        data = _extract_json(brain.invoke(_advice_prompt(req), max_tokens=1200))
        wisdom = str(data.get("wisdom") or "").strip()
        if not wisdom:
            raise ValueError("model returned no wisdom")
        focus = data.get("focus_suggestion")
        return AdviceResponse(
            wisdom=wisdom,
            focus_suggestion=str(focus).strip() if focus else None,
            source="bedrock",
        )
    except Exception as exc:
        log.warning("Bedrock advice failed, using offline cat: %s", exc)
        brain.last_error = str(exc)
        fallback = mock_advice(req)
        fallback.note = f"Offline cat answered — {exc}"
        return fallback
