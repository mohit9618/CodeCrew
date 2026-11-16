import asyncio
import selectors
import uuid
from pathlib import Path

from langgraph.types import Command

from backend.app.database.checkpoint import pool, get_checkpointer
from backend.app.graph.workflow import compile_workflow
from backend.app.services.run_service import run_workflow, get_workflow_state
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
            "Implement a factorial(n) function in calculator.py. "
"Return the factorial of a non-negative integer. "
"Raise ValueError for negative inputs. "
"Add unit tests for zero, one, normal positive values, and negative inputs."
            """,
            "repo_url": "",
            "workspace": WORKSPACE,
            "repo_tree": repo_tree,
            "file_contents": file_contents,
            "retry_count": 0,
            "max_attempts": 3,
            "revision_count": 0,
            "max_revisions": 2,
            "status": "started"
        }

        run_id = str(uuid.uuid4())

        print("\nStarting complete CodeCrew workflow...")

        result = await run_workflow(
            workflow=workflow,
            run_id=run_id,
            initial_state=initial_state
        )

        print("\nWorkflow paused for human approval.")

        state = await get_workflow_state(workflow, run_id)

        if not state or not state["next"]:
            print("Workflow did not pause as expected.")
            print("Final state:", result)
            return

        print("\nAPPROVAL DETAILS:")
        print("Interrupt:", result.get("__interrupt__"))
        print("Next nodes:", state["next"])

        decision = input(
            "\nEnter decision (approve/request_changes/reject): "
        ).strip().lower()

        feedback = ""

        if decision == "request_changes":
            feedback = input("Enter review feedback: ").strip()

        if decision not in ["approve", "request_changes", "reject"]:
            print("Invalid decision.")
            return

        print("\nResuming workflow...")

        result = await workflow.ainvoke(
            Command(resume={
                "decision": decision,
                "feedback": feedback
            }),
            config={"configurable": {"thread_id": run_id}}
        )

        print("\nFINAL STATUS:")
        print(result.get("status"))

        print("\nAPPROVAL STATUS:")
        print(result.get("approval_status"))

        print("\nREVISION COUNT:")
        print(result.get("revision_count"))

        print("\nIMPLEMENTATION SUMMARY:")
        print(result.get("implementation_summary", ""))

        print("\nAPPLIED FILES:")
        for file_path in result.get("applied_files", []):
            print("-", file_path)

        print("\nTEST REPORT:")
        print(result.get("test_report"))

        print("\nREVIEW REPORT:")
        print(result.get("review_report"))

        print("\nNEXT NODES:")
        state = await get_workflow_state(workflow, run_id)
        print(state["next"] if state else None)

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