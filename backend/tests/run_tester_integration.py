import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sandbox.execution import run_tests
from backend.app.agents.tester import TesterAgent


async def main():
    workspace = r"C:\CodeCrew\workspace"

    execution = run_tests(workspace)

    tester = TesterAgent()

    result = await tester.analyze_results(
        task="Implement calculator addition and subtraction",
        implementation_summary="Added add and subtract functions.",
        test_output=(
            execution["stdout"] + "\n" + execution["stderr"]
        ),
        exit_code=execution["exit_code"]
    )

    print("\nTEST EXECUTION:")
    print("Exit Code:", execution["exit_code"])

    print("\nTESTER AGENT ANALYSIS:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())