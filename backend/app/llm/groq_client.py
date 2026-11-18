import asyncio
import os
import time
from collections import deque

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq
from pydantic import PrivateAttr

load_dotenv()


class GroqRateLimiter:
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

        # Optional tokens-per-day tracking (Groq's free tier allows
        # 200,000 for gpt-oss-120b). Estimates only; 0 disables it.
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
        input_tokens = max(1, input_chars // 4)

        return input_tokens + self._reserved_output_tokens

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


_tpm_limit = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
_safety_margin = int(os.getenv("GROQ_TPM_SAFETY_MARGIN", "1000"))

_tpd_limit = int(os.getenv("GROQ_TPD_LIMIT", "200000"))

_shared_limiter = GroqRateLimiter(
    tpm_limit=_tpm_limit,
    safety_margin=_safety_margin,
    tpd_limit=_tpd_limit
)

_shared_clients = {}


def get_groq_llm(max_tokens: int | None = None):
    if max_tokens is None:
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "1200"))

    max_tokens = int(max_tokens)

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    if max_tokens not in _shared_clients:
        _shared_clients[max_tokens] = SharedGroqClient(
            limiter=_shared_limiter,
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
            max_tokens=max_tokens
        )

    return _shared_clients[max_tokens]
