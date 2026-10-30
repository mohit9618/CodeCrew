
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.tester import TesterAgent


async def main():
    agent = TesterAgent()

    result = await agent.analyze_results(
        task="Add a subtract function to calculator.py.",
        implementation_summary="Added subtract(a, b).",
        test_output="""
        test_add PASSED
        test_subtract PASSED
        test_multiply FAILED

        AssertionError: Expected 12, got 10

        2 passed, 1 failed
        """,
        exit_code=1
    )

    print("\nTESTER AGENT OUTPUT\n")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
