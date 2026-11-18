import asyncio

import httpx
import pytest
from groq import RateLimitError
from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq

from backend.app.agents.developer import DeveloperAgent, DeveloperOutput
from backend.app.llm.groq_client import GroqRateLimiter, SharedGroqClient
from backend.app.utils.text import clamp_text


def rate_limit_error(retry_after):
    request = httpx.Request("POST", "https://api.groq.com/x")
    response = httpx.Response(
        429,
        headers={"retry-after": str(retry_after)},
        request=request
    )
    return RateLimitError("rate limited", response=response, body=None)


def make_client():
    return SharedGroqClient(
        limiter=GroqRateLimiter(tpm_limit=8000, safety_margin=1000),
        model="openai/gpt-oss-120b",
        api_key="test-key"
    )


async def test_long_retry_after_fails_fast_instead_of_sleeping(monkeypatch):
    calls = []
    slept = []

    async def fake_ainvoke(self, *args, **kwargs):
        calls.append(1)
        raise rate_limit_error(3600)  # a daily limit, not a per-minute one

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ChatGroq, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="daily token"):
        await make_client().ainvoke("hello")

    assert len(calls) == 1
    assert slept == []


async def test_short_retry_after_waits_and_retries(monkeypatch):
    calls = []
    slept = []

    async def fake_ainvoke(self, *args, **kwargs):
        calls.append(1)

        if len(calls) == 1:
            raise rate_limit_error(2)

        return "ok"

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(ChatGroq, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    assert await make_client().ainvoke("hello") == "ok"
    assert len(calls) == 2
    assert slept == [2.0]


async def test_daily_usage_warning(capsys):
    limiter = GroqRateLimiter(
        tpm_limit=100_000,
        safety_margin=0,
        tpd_limit=1_000
    )

    await limiter.acquire(500)
    assert "WARNING" not in capsys.readouterr().out

    await limiter.acquire(400)  # 900 >= 80% of 1,000
    assert "WARNING" in capsys.readouterr().out

    await limiter.acquire(50)  # already warned: stay quiet
    assert "WARNING" not in capsys.readouterr().out


# --------------------------------------------------------------- developer

class FlakyStructuredLLM:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.last_messages = None

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_messages = messages

        if self.calls <= self.failures:
            raise OutputParserException("truncated JSON")

        return DeveloperOutput(summary="ok", changes=[])


def make_developer(llm):
    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.specialization = "backend"
    agent.structured_llm = llm
    return agent


async def test_developer_retries_once_on_invalid_json():
    llm = FlakyStructuredLLM(failures=1)

    result = await make_developer(llm).generate_changes("t", "tree", "files")

    assert result.summary == "ok"
    assert llm.calls == 2
    assert "not valid JSON" in llm.last_messages[-1][1]


async def test_developer_gives_up_after_retry_budget():
    llm = FlakyStructuredLLM(failures=5)

    with pytest.raises(OutputParserException):
        await make_developer(llm).generate_changes("t", "tree", "files")

    assert llm.calls == 2  # first try + one retry


# -------------------------------------------------------------------- text

def test_clamp_text_keeps_tail_when_asked():
    text = "A" * 100 + "FAILED"

    clipped = clamp_text(text, 20, "output", keep="tail")

    assert clipped.endswith("FAILED")
    assert "truncated" in clipped


def test_clamp_text_accepts_non_strings():
    assert clamp_text({"a": 1}, 100, "spec") == "{'a': 1}"
    assert clamp_text(None, 10, "x") == ""
