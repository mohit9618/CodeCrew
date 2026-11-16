
import asyncio

from backend.app.llm import get_groq_llm


async def main():
    llm = get_groq_llm()

    print("Testing shared Groq client...\n")

    print("Test 1: First request")
    response1 = await llm.ainvoke(
        "Reply with exactly: CodeCrew Groq connection successful."
    )
    print("Response:", response1.content)

    print("\nTest 2: Second request")
    response2 = await llm.ainvoke(
        "Reply with exactly: Shared rate limiter working."
    )
    print("Response:", response2.content)

    print("\nTest 3: Shared client verification")
    llm2 = get_groq_llm()

    print("Same client instance:", llm is llm2)
    print(
        "Same rate limiter:",
        llm._rate_limiter is llm2._rate_limiter
    )

    print("\nAll tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
