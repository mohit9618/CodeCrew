
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.checkpoint.memory import InMemorySaver

from backend.app.graph.workflow import compile_workflow
from backend.app.mcp.clients.filesystem import FilesystemClient


WORKSPACE = r"C:\CodeCrew\workspace"


async def main():
    filesystem = FilesystemClient(workspace=WORKSPACE)

    await filesystem.connect()

    try:
        repo_tree = await filesystem.get_directory_tree()

        file_contents = await filesystem.read_multiple_files([
            str(Path(WORKSPACE) / "calculator.py"),
            str(Path(WORKSPACE) / "tests" / "test_calculator.py")
        ])

    finally:
        await filesystem.disconnect()

    initial_state = {
        "request": """
        Add a divide(a, b) function to calculator.py.
        Add tests for positive, negative and zero division.
        Handle division by zero appropriately.
        Preserve existing functionality.
        """,
        "workspace": WORKSPACE,
        "repo_tree": repo_tree,
        "file_contents": file_contents,
        "retry_count": 0,
        "max_attempts": 3,
        "status": "running",
        "last_error": ""
    }

    print("\nStarting CodeCrew LangGraph workflow...\n")

    app = compile_workflow(InMemorySaver())

    result = await app.ainvoke(
        initial_state,
        {
            "configurable": {"thread_id": "manual-test"},
            "recursion_limit": 100
        }
    )

    print("\nFINAL WORKFLOW RESULT")
    print("Status:", result.get("status"))
    print("Attempts:", result.get("retry_count"))
    print("Applied Files:", result.get("applied_files"))
    print("Test Report:", result.get("test_report"))


if __name__ == "__main__":
    asyncio.run(main())
