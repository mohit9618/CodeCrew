import asyncio
from backend.app.llm.groq_client import GroqRateLimiter


async def main():
    limiter = GroqRateLimiter(
        tpm_limit=100,
        safety_margin=0
    )

    await limiter.acquire(60)
    print("First reservation successful")

    await limiter.acquire(30)
    print("Second reservation successful")

    print("Used tokens:", sum(tokens for _, tokens in limiter.requests))
    print("Available tokens:", limiter.max_budget - sum(
        tokens for _, tokens in limiter.requests
    ))


if __name__ == "__main__":
    asyncio.run(main())