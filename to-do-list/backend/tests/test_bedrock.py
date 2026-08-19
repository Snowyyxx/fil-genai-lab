"""Tests for the Bedrock layer, using a stubbed boto3 client.

These cover the parts that can't be exercised without an AWS account: the
Converse request shape, response parsing, open-weight filtering and the
JSON-repair path. Run with the rest of the suite:

    cd backend && MEOW_FORCE_MOCK=1 pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zen_cat
from schemas import AdviceRequest, BreakdownRequest


# --------------------------------------------------------------------------- #
# id parsing / ranking
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("meta.llama3-3-70b-instruct-v1:0", "meta"),
        ("us.meta.llama3-3-70b-instruct-v1:0", "meta"),
        ("eu.mistral.mistral-small-2402-v1:0", "mistral"),
        ("deepseek.r1-v1:0", "deepseek"),
        ("us.deepseek.r1-v1:0", "deepseek"),
        ("openai.gpt-oss-120b-1:0", "openai"),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "anthropic"),
        ("amazon.nova-lite-v1:0", "amazon"),
    ],
)
def test_provider_ignores_regional_prefix(model_id, expected):
    assert zen_cat._provider(model_id) == expected


def test_rank_puts_the_most_capable_open_model_first():
    ranked = zen_cat._rank(
        [
            "mistral.mistral-7b-instruct-v0:2",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "openai.gpt-oss-120b-1:0",
            "meta.llama3-8b-instruct-v1:0",
        ]
    )
    assert ranked[0] == "openai.gpt-oss-120b-1:0"
    assert ranked[1] == "us.meta.llama3-3-70b-instruct-v1:0"
    assert ranked[-1] == "mistral.mistral-7b-instruct-v0:2"


def test_rank_dedupes():
    assert zen_cat._rank(["deepseek.r1-v1:0", "deepseek.r1-v1:0"]) == ["deepseek.r1-v1:0"]


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def test_extract_json_handles_fences_prose_and_think_blocks():
    assert zen_cat._extract_json('{"a": 1}') == {"a": 1}
    assert zen_cat._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert zen_cat._extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}
    # DeepSeek-R1 style reasoning prefix
    assert zen_cat._extract_json('<think>hmm, the user wants {"b": 2}</think>{"a": 1}') == {"a": 1}


def test_extract_json_raises_when_there_is_no_object():
    with pytest.raises(ValueError):
        zen_cat._extract_json("I would rather nap than emit JSON.")


# --------------------------------------------------------------------------- #
# Converse plumbing
# --------------------------------------------------------------------------- #
class FakeRuntime:
    """Stands in for boto3's bedrock-runtime client."""

    def __init__(self, text="{}", fail_on_system=False):
        self.text = text
        self.fail_on_system = fail_on_system
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_system and "system" in kwargs:
            raise RuntimeError("ValidationException: this model does not support system messages")
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": self.text}]}},
            "stopReason": "end_turn",
        }


@pytest.fixture()
def brain(monkeypatch):
    """A CatBrain wired to a fake runtime, with discovery short-circuited."""
    b = zen_cat.CatBrain()
    fake = FakeRuntime()
    b._runtime, b._control = fake, object()
    b._model_id, b._resolved = "us.meta.llama3-3-70b-instruct-v1:0", True
    monkeypatch.setattr(zen_cat, "brain", b)
    monkeypatch.setattr(zen_cat.settings, "force_mock", False)
    return b


def test_invoke_sends_a_well_formed_converse_request(brain):
    brain._runtime.text = "purr"
    assert brain.invoke("hello", max_tokens=64) == "purr"

    (call,) = brain._runtime.calls
    assert call["modelId"] == "us.meta.llama3-3-70b-instruct-v1:0"
    assert call["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert call["system"][0]["text"].startswith("You are Meowstermind")
    assert call["inferenceConfig"]["maxTokens"] == 64


def test_invoke_retries_without_system_block_when_the_model_rejects_it(brain):
    brain._runtime = FakeRuntime(text="purr", fail_on_system=True)
    assert brain.invoke("hello") == "purr"

    first, second = brain._runtime.calls
    assert "system" in first
    assert "system" not in second
    # the persona must survive, folded into the user turn
    assert "You are Meowstermind" in second["messages"][0]["content"][0]["text"]
    assert "hello" in second["messages"][0]["content"][0]["text"]


def test_invoke_rejects_an_empty_reply(brain):
    brain._runtime.text = "   "
    with pytest.raises(RuntimeError, match="empty reply"):
        brain.invoke("hello")


def test_breakdown_parses_a_model_reply(brain):
    brain._runtime.text = """{"subtasks": [
        {"title": "Open the document", "estimate_minutes": 5},
        {"title": "Draft the outline", "estimate_minutes": 30}
    ], "zen_comment": "One paw at a time."}"""

    result = zen_cat.breakdown(BreakdownRequest(title="Write the report"), "Write the report")
    assert result.source == "bedrock"
    assert [s.title for s in result.subtasks] == ["Open the document", "Draft the outline"]
    assert result.subtasks[1].estimate_minutes == 30
    assert result.zen_comment == "One paw at a time."
    assert result.note is None


def test_breakdown_falls_back_when_the_model_returns_junk(brain):
    brain._runtime.text = "I'm just a cat, I don't do JSON."
    result = zen_cat.breakdown(BreakdownRequest(title="Write the report"), "Write the report")
    assert result.source == "mock"
    assert result.note and "Offline cat answered" in result.note
    assert result.subtasks  # the user still gets a usable breakdown


def test_breakdown_respects_max_subtasks(brain):
    brain._runtime.text = (
        '{"subtasks": ['
        + ",".join(f'{{"title": "step {i}", "estimate_minutes": 5}}' for i in range(8))
        + '], "zen_comment": "ok"}'
    )
    result = zen_cat.breakdown(
        BreakdownRequest(title="Big task", max_subtasks=3), "Big task"
    )
    assert len(result.subtasks) == 3


def test_advice_parses_a_model_reply(brain):
    brain._runtime.text = '{"wisdom": "Begin softly.", "focus_suggestion": "Start with the plants."}'
    result = zen_cat.advice(AdviceRequest())
    assert result.source == "bedrock"
    assert result.wisdom == "Begin softly."
    assert result.focus_suggestion == "Start with the plants."


def test_advice_falls_back_on_error(brain):
    brain._runtime.text = "no json here"
    result = zen_cat.advice(AdviceRequest())
    assert result.source == "mock"
    assert result.note and "Offline cat answered" in result.note


# --------------------------------------------------------------------------- #
# discovery filtering
# --------------------------------------------------------------------------- #
class FakeControl:
    """Stands in for boto3's bedrock (control-plane) client."""

    def __init__(self, models, profiles):
        self._models = models
        self._profiles = profiles

    def get_paginator(self, _name):
        profiles = self._profiles

        class P:
            def paginate(self):
                return [{"inferenceProfileSummaries": profiles}]

        return P()

    def list_foundation_models(self, **_kwargs):
        return {"modelSummaries": self._models}


def test_discovery_keeps_open_weights_and_drops_proprietary(monkeypatch):
    b = zen_cat.CatBrain()
    b._runtime = object()
    b._control = FakeControl(
        models=[
            {"modelId": "meta.llama3-3-70b-instruct-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "amazon.nova-lite-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
            # profile-only model: must not show up in the on-demand list
            {"modelId": "openai.gpt-oss-120b-1:0", "inferenceTypesSupported": ["INFERENCE_PROFILE"]},
            # image model: wrong modality
            {"modelId": "meta.some-image-model-v1:0", "inferenceTypesSupported": ["ON_DEMAND"],
             "outputModalities": ["IMAGE"]},
        ],
        profiles=[
            {"inferenceProfileId": "us.openai.gpt-oss-120b-1:0"},
            {"inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
        ],
    )
    monkeypatch.setattr(b, "_clients", lambda: (b._runtime, b._control))

    found = b.available_models()
    assert found["on_demand_models"] == ["meta.llama3-3-70b-instruct-v1:0"]
    assert found["inference_profiles"] == ["us.openai.gpt-oss-120b-1:0"]
    assert found["errors"] == []

    # profiles win: gpt-oss-120b outranks llama3-3-70b
    assert b.resolve_model_id() == "us.openai.gpt-oss-120b-1:0"


def test_discovery_reports_why_it_failed(monkeypatch):
    b = zen_cat.CatBrain()

    def boom():
        raise RuntimeError("Unable to locate credentials")

    monkeypatch.setattr(b, "_clients", boom)
    found = b.available_models()
    assert found["inference_profiles"] == [] and found["on_demand_models"] == []
    assert "Unable to locate credentials" in found["errors"][0]

    assert b.resolve_model_id() is None
    assert "could not reach Bedrock" in b.last_error
