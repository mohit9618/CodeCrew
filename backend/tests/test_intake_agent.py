
import asyncio

from backend.app.agents.intake import IntakeAgent


async def main():
    agent = IntakeAgent()

    result = await agent.analyze_request(
        request="Add a power function to calculator.py.",
        repo_url="https://github.com/example/example",
        repo_tree="calculator.py\ntests/test_calculator.py",
        file_contents="""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
