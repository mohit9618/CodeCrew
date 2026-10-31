
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.agents.developer import DeveloperAgent
from backend.app.agents.tester import TesterAgent
from backend.app.mcp.clients.filesystem import FilesystemClient
from sandbox.execution import run_tests


WORKSPACE = r"C:\CodeCrew\workspace"
MAX_ATTEMPTS = 3


async def run_development_loop(task):
    filesystem = FilesystemClient(workspace=WORKSPACE)
    developer = DeveloperAgent()
    tester = TesterAgent()

    last_failure = ""

    await filesystem.connect()

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"\n{'=' * 15} ATTEMPT {attempt} {'=' * 15}")

            repo_tree = await filesystem.get_directory_tree()

            file_contents = await filesystem.read_multiple_files([
                str(Path(WORKSPACE) / "calculator.py"),
                str(Path(WORKSPACE) / "tests" / "test_calculator.py")
            ])

            current_task = task

            if last_failure:
                current_task += f"""

                PREVIOUS TEST FAILURE:
                {last_failure}

                Fix the implementation based on the failure.
                Do not remove or weaken existing tests just to
                make them pass.
                """

            print("\nGenerating code changes...")

            output = await developer.generate_changes(
                task=current_task,
                repo_tree=repo_tree,
                file_contents=file_contents
            )

            if output.questions:
                print("\nDeveloper needs clarification:")
                for question in output.questions:
                    print("-", question)
                return {
                    "status": "needs_clarification",
                    "attempts": attempt
                }

            if not output.changes:
                print("\nDeveloper generated no changes.")
                return {
                    "status": "no_changes",
                    "attempts": attempt
                }

            print("\nImplementation:", output.summary)

            applied_files = await developer.apply_changes(
                output,
                filesystem
            )

            print("\nApplied files:")
            for file_path in applied_files:
                print("-", file_path)

            print("\nRunning tests inside Docker...")

            execution = run_tests(WORKSPACE)

            test_output = (
                execution["stdout"] + "\n" + execution["stderr"]
            )

            print("\nDocker exit code:", execution["exit_code"])

            analysis = await tester.analyze_results(
                task=task,
                implementation_summary=output.summary,
                test_output=test_output,
                exit_code=execution["exit_code"]
            )

            print("\nTester Analysis:")
            print(analysis.model_dump_json(indent=2))

            if (
                execution["exit_code"] == 0
                and not execution["timed_out"]
                and analysis.overall_status == "passed"
                and analysis.failed_count == 0
            ):
                print("\nAll tests passed!")
                return {
                    "status": "passed",
                    "attempts": attempt,
                    "applied_files": applied_files,
                    "test_report": analysis.model_dump()
                }

            last_failure = f"""
            EXIT CODE: {execution['exit_code']}
            TIMED OUT: {execution['timed_out']}

            TEST OUTPUT:
            {test_output}

            TESTER ANALYSIS:
            {analysis.model_dump_json(indent=2)}
            """

            print("\nTests failed. Sending feedback to Developer Agent...")

        print("\nMaximum attempts reached.")

        return {
            "status": "failed",
            "attempts": MAX_ATTEMPTS,
            "last_failure": last_failure
        }

    finally:
        await filesystem.disconnect()


async def main():
    task = """
    Add a multiply(a, b) function to calculator.py.
    Add appropriate tests for positive, negative, and zero values
    in tests/test_calculator.py.
    Preserve the existing add and subtract functions and tests.
    """

    result = await run_development_loop(task)

    print("\nFINAL RESULT:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
