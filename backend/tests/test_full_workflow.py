
import asyncio
import selectors
import uuid
from pathlib import Path

from backend.app.database.checkpoint import pool, get_checkpointer
from backend.app.graph.workflow import compile_workflow
from backend.app.services.run_service import run_workflow
from backend.app.mcp.clients.filesystem import FilesystemClient


WORKSPACE = r"C:\CodeCrew\workspace"


async def main():
    filesystem = FilesystemClient(workspace=WORKSPACE)

    await pool.open()
    await filesystem.connect()

    try:
        checkpointer = get_checkpointer()
        await checkpointer.setup()

        workflow = compile_workflow(checkpointer)

        repo_tree = await filesystem.get_directory_tree()

        file_contents = await filesystem.read_multiple_files([
            str(Path(WORKSPACE) / "calculator.py"),
            str(Path(WORKSPACE) / "tests" / "test_calculator.py")
        ])

        initial_state = {
            "request": """
            Add a cube(a) function to calculator.py.
            Add tests for positive, negative, and zero values.
            Preserve all existing functions and tests.
            """,
            "repo_url": "",
            "workspace": WORKSPACE,
            "repo_tree": repo_tree,
            "file_contents": file_contents,
            "retry_count": 0,
            "max_attempts": 3,
            "status": "started"
        }

        run_id = str(uuid.uuid4())

        print("\nStarting full CodeCrew workflow...")

        result = await run_workflow(
            workflow=workflow,
            run_id=run_id,
            initial_state=initial_state
        )

        print("\nARCHITECT TASKS:")
        for task in result.get("tasks", []):
            print(
                task.get("task_id"),
                task.get("title"),
                task.get("assigned_developer")
            )

        print("\nEXECUTION BATCHES:")
        print(result.get("execution_batches"))

        print("\nFINAL DEVELOPER STATE:")
        print("Stage:", result.get("developer_stage"))
        print("Summaries:", result.get("stage_summary"))
        print("Applied files:", result.get("applied_files"))

        print("\nFINAL STATUS:")
        print(result.get("status"))

        print("\nDEVELOPER STAGE:")
        print(result.get("developer_stage"))

        print("\nIMPLEMENTATION SUMMARY:")
        print(result.get("implementation_summary", ""))

        print("\nSTAGE SUMMARIES:")
        for summary in result.get("stage_summary", []):
            print("-", summary)

        print("\nAPPLIED FILES:")
        for file_path in result.get("applied_files", []):
            print("-", file_path)

        print("\nTEST REPORT:")
        print(result.get("test_report"))

    finally:
        await filesystem.disconnect()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
