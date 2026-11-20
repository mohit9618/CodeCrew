"""Per-role model choice, per-model rate limits and connection retries."""
import asyncio

import httpx
import pytest
from groq import APIConnectionError
from langchain_groq import ChatGroq

from backend.app.llm import groq_client
from backend.app.llm.groq_client import (
    DEFAULT_MODEL,
    GroqRateLimiter,
    SharedGroqClient,
    get_groq_llm,
    get_limiter,
    resolve_model,
    tpm_for_model,
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for name in ("GROQ_MODEL", "GROQ_TPM_LIMIT", "GROQ_MODEL_ARCHITECT",
                 "GROQ_MODEL_BACKEND_DEVELOPER"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_client, "_limiters", {})
    monkeypatch.setattr(groq_client, "_shared_clients", {})


# ------------------------------------------------------------ model choice

def test_default_model_is_used_when_nothing_is_configured():
    assert resolve_model("architect") == DEFAULT_MODEL
    assert resolve_model() == DEFAULT_MODEL


def test_global_model_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3-32b")

    assert resolve_model("architect") == "qwen/qwen3-32b"
    assert resolve_model("tester") == "qwen/qwen3-32b"


def test_role_specific_model_wins(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3-32b")
    monkeypatch.setenv("GROQ_MODEL_ARCHITECT", "openai/gpt-oss-120b")

    assert resolve_model("architect") == "openai/gpt-oss-120b"
    assert resolve_model("reviewer") == "qwen/qwen3-32b"


# ------------------------------------------------------------- rate limits

def test_each_model_has_its_own_limiter_and_budget():
    first = get_limiter("llama-3.3-70b-versatile")
    second = get_limiter("openai/gpt-oss-120b")

    assert first is get_limiter("llama-3.3-70b-versatile")
    assert first is not second
    assert first.tpm_limit == 12_000 and second.tpm_limit == 8_000


def test_unknown_models_get_a_conservative_budget():
    assert tpm_for_model("some/new-model") == 6_000


def test_tpm_override_applies_to_all_models(monkeypatch):
    monkeypatch.setenv("GROQ_TPM_LIMIT", "20000")

    assert tpm_for_model("llama-3.3-70b-versatile") == 20_000
    assert tpm_for_model("some/new-model") == 20_000


def test_clients_are_cached_per_model_and_output_size(monkeypatch):
    a = get_groq_llm(max_tokens=500, role="architect")

    assert get_groq_llm(max_tokens=500, role="architect") is a
    assert get_groq_llm(max_tokens=600, role="architect") is not a

    monkeypatch.setenv("GROQ_MODEL_ARCHITECT", "openai/gpt-oss-120b")
    assert get_groq_llm(max_tokens=500, role="architect") is not a


def test_prompt_char_budget_follows_the_model_budget():
    client = SharedGroqClient(
        limiter=GroqRateLimiter(tpm_limit=12_000, safety_margin=1_000),
        model="m", api_key="k", max_tokens=3_000,
    )

    # (11,000 budget - 3,000 output) tokens * 4 chars
    assert client.prompt_char_budget() == 32_000
    assert client.prompt_char_budget(overhead_tokens=1_000) == 28_000
    assert client.prompt_char_budget(overhead_tokens=50_000) == 0


# ------------------------------------------------------ connection retries

def connection_error():
    return APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com/x")
    )


def make_client():
    return SharedGroqClient(
        limiter=GroqRateLimiter(tpm_limit=8000, safety_margin=1000),
        model="m", api_key="test-key",
    )


async def test_dropped_connection_is_retried(monkeypatch):
    calls, slept = [], []

    async def flaky(self, *args, **kwargs):
        calls.append(1)

        if len(calls) < 3:
            raise connection_error()

        return "ok"

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ChatGroq, "ainvoke", flaky)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert await make_client().ainvoke("hi") == "ok"
    assert len(calls) == 3
    assert slept == [2, 4]          # backs off between attempts


async def test_connection_errors_give_up_after_the_retry_budget(monkeypatch):
    calls = []

    async def always_down(self, *args, **kwargs):
        calls.append(1)
        raise connection_error()

    async def fake_sleep(seconds):
        pass

    monkeypatch.setenv("GROQ_MAX_RETRIES", "2")
    monkeypatch.setattr(ChatGroq, "ainvoke", always_down)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(APIConnectionError):
        await make_client().ainvoke("hi")

    assert len(calls) == 3          # first try + 2 retries
