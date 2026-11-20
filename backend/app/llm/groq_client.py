"""Groq chat client with rate limiting, retries and per-role model choice.

Groq's free tier limits tokens per minute (TPM) *per model*, so every model
gets its own limiter. Which model an agent uses is configurable per role:

    GROQ_MODEL                 default model for every role
    GROQ_MODEL_<ROLE>          override for one role, e.g. GROQ_MODEL_ARCHITECT
    GROQ_TPM_LIMIT             override the TPM budget for every model
    GROQ_TPM_SAFETY_MARGIN     tokens kept free below the limit (default 1000)
    GROQ_TPD_LIMIT             optional daily token warning threshold
    GROQ_MAX_RETRIES           retries for 429s and connection errors
    GROQ_MAX_RETRY_WAIT        longest retry-after (seconds) worth waiting for

Roles: intake, project_manager, architect, backend_developer,
database_developer, frontend_developer, tester, reviewer, analyzer.
"""
import asyncio
import os
import time
from collections import deque

from dotenv import load_dotenv
from groq import APIConnectionError, RateLimitError
from langchain_groq import ChatGroq
from pydantic import PrivateAttr

load_dotenv()

# Default for every role. A non-reasoning model: reasoning models spend part
# of max_tokens on hidden thinking, which truncates long JSON answers.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Free-tier tokens-per-minute per model (Groq rate-limits page). Models not
# listed fall back to DEFAULT_TPM; set GROQ_TPM_LIMIT to override.
MODEL_TPM = {
    "llama-3.3-70b-versatile": 12_000,
    "llama-3.1-8b-instant": 6_000,
    "openai/gpt-oss-120b": 8_000,
    "openai/gpt-oss-20b": 8_000,
    "qwen/qwen3-32b": 6_000,
    "moonshotai/kimi-k2-instruct": 10_000,
    "meta-llama/llama-4-scout-17b-16e-instruct": 30_000,
}
DEFAULT_TPM = 6_000

CHARS_PER_TOKEN = 4


class GroqRateLimiter:
    """Sliding one-minute token budget shared by all calls to one model."""

    def __init__(
        self,
        tpm_limit: int = 8000,
        safety_margin: int = 1000,
        tpd_limit: int = 0
    ):
        self.tpm_limit = tpm_limit
        self.max_budget = tpm_limit - safety_margin
        self.requests = deque()
        self.lock = asyncio.Lock()

        # Optional tokens-per-day tracking. Estimates only; 0 disables it.
        self.tpd_limit = tpd_limit
        self.daily_requests = deque()
        self._daily_warned = False

    def record_daily_usage(self, token_estimate: int):
        if not self.tpd_limit:
            return

        now = time.time()

        while (
            self.daily_requests
            and now - self.daily_requests[0][0] >= 86_400
        ):
            self.daily_requests.popleft()

        self.daily_requests.append((now, token_estimate))

        used = sum(tokens for _, tokens in self.daily_requests)

        if used >= self.tpd_limit * 0.8 and not self._daily_warned:
            self._daily_warned = True
            print(
                f"[Groq Rate Limiter] WARNING: about {used:,} of "
                f"{self.tpd_limit:,} daily tokens used (estimate). "
                "Further calls may fail with a daily-limit 429."
            )
        elif used < self.tpd_limit * 0.8:
            self._daily_warned = False

    async def acquire(self, token_estimate: int):
        if token_estimate > self.max_budget:
            raise ValueError(
                f"Request requires approximately {token_estimate} tokens, "
                f"but the safe TPM budget is {self.max_budget}. "
                "Reduce the prompt or output token limit."
            )

        async with self.lock:
            while True:
                curr_time = time.monotonic()

                while (
                    self.requests
                    and curr_time - self.requests[0][0] >= 60
                ):
                    self.requests.popleft()

                used_tokens = sum(
                    tokens for _, tokens in self.requests
                )

                available_tokens = self.max_budget - used_tokens

                if token_estimate <= available_tokens:
                    self.requests.append(
                        (curr_time, token_estimate)
                    )
                    self.record_daily_usage(token_estimate)
                    return

                oldest_time = self.requests[0][0]
                wait_time = max(
                    0.1,
                    60 - (curr_time - oldest_time)
                )

                print(
                    f"[Groq Rate Limiter] Waiting {wait_time:.1f}s "
                    f"| Used: {used_tokens} "
                    f"| Requested: {token_estimate} "
                    f"| Budget: {self.max_budget}"
                )

                await asyncio.sleep(wait_time)


class SharedGroqClient(ChatGroq):
    _rate_limiter: GroqRateLimiter = PrivateAttr()
    _reserved_output_tokens: int = PrivateAttr()
    _max_retries: int = PrivateAttr()
    _max_retry_wait: float = PrivateAttr()

    def __init__(self, limiter: GroqRateLimiter, **kwargs):
        output_tokens = kwargs.get("max_tokens")
        if output_tokens is None:
            output_tokens = int(os.getenv("GROQ_MAX_TOKENS", "1200"))

        kwargs["max_tokens"] = int(output_tokens)

        super().__init__(**kwargs)

        self._rate_limiter = limiter
        self._reserved_output_tokens = int(output_tokens)
        self._max_retries = int(
            os.getenv("GROQ_MAX_RETRIES", "4")
        )
        self._max_retry_wait = float(
            os.getenv("GROQ_MAX_RETRY_WAIT", "120")
        )

    def estimate_tokens(self, input_data) -> int:
        input_chars = len(str(input_data))
        input_tokens = max(1, input_chars // CHARS_PER_TOKEN)

        return input_tokens + self._reserved_output_tokens

    def prompt_char_budget(self, overhead_tokens: int = 0) -> int:
        """Characters of prompt that fit this model's per-minute budget.

        `overhead_tokens` is for fixed text (system prompt) the caller
        already knows it will send.
        """
        tokens = (
            self._rate_limiter.max_budget
            - self._reserved_output_tokens
            - overhead_tokens
        )

        return max(0, tokens * CHARS_PER_TOKEN)

    def get_retry_delay(self, error, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}

        retry_after = headers.get("retry-after")

        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except (TypeError, ValueError):
                pass

        return min(2 ** (attempt + 1), 30)

    async def ainvoke(self, input, config=None, **kwargs):
        token_estimate = self.estimate_tokens(input)

        await self._rate_limiter.acquire(token_estimate)

        for attempt in range(self._max_retries + 1):
            try:
                return await super().ainvoke(
                    input,
                    config=config,
                    **kwargs
                )

            except RateLimitError as error:
                if attempt == self._max_retries:
                    raise

                wait_time = self.get_retry_delay(error, attempt)

                # A short retry-after is a per-minute limit: wait it out.
                # A long one (minutes to hours) means the daily token
                # limit is used up. Sleeping that long would just hang
                # the run, so fail with a clear message instead.
                if wait_time > self._max_retry_wait:
                    raise RuntimeError(
                        f"Groq asked to wait {wait_time:.0f}s before the "
                        "next request, which usually means the daily token "
                        "limit is exhausted. Try again later, use a model "
                        "or key with a higher limit, or lower "
                        "GROQ_TPM_LIMIT / prompt sizes."
                    ) from error

                print(
                    f"[Groq 429] Retrying in {wait_time:.1f}s "
                    f"({attempt + 1}/{self._max_retries})"
                )

                await asyncio.sleep(wait_time)

            except APIConnectionError as error:
                # Covers DNS failures, dropped connections and timeouts
                # (APITimeoutError is a subclass).
                if attempt == self._max_retries:
                    raise

                wait_time = self.get_retry_delay(error, attempt)

                print(
                    f"[Groq connection] Retrying in {wait_time:.1f}s "
                    f"({attempt + 1}/{self._max_retries})"
                )

                await asyncio.sleep(wait_time)


def resolve_model(role: str | None = None) -> str:
    """Model for a role: GROQ_MODEL_<ROLE>, then GROQ_MODEL, then default."""
    if role:
        specific = os.getenv(f"GROQ_MODEL_{role.upper()}")

        if specific:
            return specific

    return os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def tpm_for_model(model: str) -> int:
    override = os.getenv("GROQ_TPM_LIMIT")

    if override:
        return int(override)

    return MODEL_TPM.get(model, DEFAULT_TPM)


_limiters: dict[str, GroqRateLimiter] = {}
_shared_clients: dict[tuple[str, int], SharedGroqClient] = {}


def get_limiter(model: str) -> GroqRateLimiter:
    """One limiter per model, because Groq limits each model separately."""
    if model not in _limiters:
        _limiters[model] = GroqRateLimiter(
            tpm_limit=tpm_for_model(model),
            safety_margin=int(os.getenv("GROQ_TPM_SAFETY_MARGIN", "1000")),
            tpd_limit=int(os.getenv("GROQ_TPD_LIMIT", "0"))
        )

    return _limiters[model]


def get_groq_llm(
    max_tokens: int | None = None,
    role: str | None = None,
    model: str | None = None
) -> SharedGroqClient:
    if max_tokens is None:
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "1200"))

    max_tokens = int(max_tokens)

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    model = model or resolve_model(role)

    # Never reserve more than half of the model's per-minute budget for the
    # answer: on a small-limit model a large output reservation would leave
    # no room for the prompt (and the rate limiter would reject it).
    max_tokens = min(max_tokens, max(256, get_limiter(model).max_budget // 2))

    key = (model, max_tokens)

    if key not in _shared_clients:
        _shared_clients[key] = SharedGroqClient(
            limiter=get_limiter(model),
            model=model,
            temperature=0,
            max_tokens=max_tokens
        )

    return _shared_clients[key]
