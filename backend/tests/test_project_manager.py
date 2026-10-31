
import asyncio

from backend.app.agents.intake import IntakeAgent
from backend.app.agents.project_manager import ProjectManagerAgent


async def main():
    intake = IntakeAgent()
    project_manager = ProjectManagerAgent()

    intake_result = await intake.analyze_request(
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

    specification = await project_manager.create_specification(
        intake_result.model_dump()
    )

    print(specification.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
