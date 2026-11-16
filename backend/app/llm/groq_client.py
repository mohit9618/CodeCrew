
import asyncio
import os
import time
from collections import deque

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_groq import ChatGroq


load_dotenv()


class GroqRateLimiter:
    def __init__(
        self,
        tpm_limit: int = 8000,
        safety_margin: int = 1000
    ):
        self.tpm_limit = tpm_limit
        self.max_budget = tpm_limit - safety_margin
        self.requests = deque()
        self.lock = asyncio.Lock()

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
    def __init__(self, limiter: GroqRateLimiter, **kwargs):
        self.rate_limiter = limiter
        self.reserved_output_tokens = int(
            os.getenv("GROQ_MAX_TOKENS", "1200")
        )
        self.max_retries = int(
            os.getenv("GROQ_MAX_RETRIES", "4")
        )

        super().__init__(**kwargs)

    def estimate_tokens(self, input_data) -> int:
        """
        Conservative character-based estimate.
        This is not an exact tokenizer.
        """
        input_chars = len(str(input_data))
        input_tokens = max(1, input_chars // 4)

        return input_tokens + self.reserved_output_tokens

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

        await self.rate_limiter.acquire(token_estimate)

        for attempt in range(self.max_retries + 1):
            try:
                return await super().ainvoke(
                    input,
                    config=config,
                    **kwargs
                )

            except RateLimitError as error:
                if attempt == self.max_retries:
                    raise

                wait_time = self.get_retry_delay(error, attempt)

                print(
                    f"[Groq 429] Rate limit reached. "
                    f"Retrying in {wait_time:.1f}s "
                    f"({attempt + 1}/{self.max_retries})"
                )

                await asyncio.sleep(wait_time)


_tpm_limit = int(os.getenv("GROQ_TPM_LIMIT", "8000"))
_safety_margin = int(
    os.getenv("GROQ_TPM_SAFETY_MARGIN", "1000")
)

_shared_limiter = GroqRateLimiter(
    tpm_limit=_tpm_limit,
    safety_margin=_safety_margin
)

_shared_client = None


def get_groq_llm():
    global _shared_client

    if _shared_client is None:
        _shared_client = SharedGroqClient(
            limiter=_shared_limiter,
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-120b"
            ),
            temperature=0,
            max_tokens=int(
                os.getenv("GROQ_MAX_TOKENS", "1200")
            )
        )

    return _shared_client
