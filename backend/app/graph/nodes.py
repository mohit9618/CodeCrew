
import asyncio

from backend.app.agents.intake import IntakeAgent
from backend.app.agents.project_manager import ProjectManagerAgent
from backend.app.agents.architect import ArchitectAgent
from backend.app.agents.developer import DeveloperAgent
from backend.app.agents.tester import TesterAgent

from backend.app.mcp.clients.filesystem import FilesystemClient
from sandbox.execution import run_tests


async def intake_node(state: dict) -> dict:
    intake = IntakeAgent()

    output = await intake.analyze_request(
        request=state["request"],
        repo_url=state["repo_url"],
        repo_tree=state["repo_tree"],
        file_contents=state["file_contents"]
    )

    return {
        "intake_result": output.model_dump(),
        "status": "intake_completed"
    }


async def project_manager_node(state: dict) -> dict:
    project_manager = ProjectManagerAgent()

    output = await project_manager.create_specification(
        state["intake_result"]
    )

    return {
        "project_spec": output.model_dump(),
        "status": "specification_completed"
    }


async def architect_node(state: dict) -> dict:
    architect = ArchitectAgent()

    output = await architect.design_architecture(
        project_spec=state["project_spec"],
        repo_tree=state["repo_tree"],
        file_contents=state["file_contents"]
    )

    architecture = output.model_dump()

    return {
        "architecture": architecture,
        "tasks": architecture["tasks"],
        "execution_batches": architecture["execution_batches"],
        "status": "architecture_completed"
    }


async def developer_node(state: dict) -> dict:
    developer = DeveloperAgent()

    attempt = state.get("retry_count", 0) + 1
    task = state["request"]

    if state.get("last_error"):
        task += f"""

        PREVIOUS TEST FAILURE:
        {state['last_error']}

        Fix the implementation based on the failure.
        Preserve existing functionality and tests.
        """

    output = await developer.generate_changes(
        task=task,
        repo_tree=state["repo_tree"],
        file_contents=state["file_contents"]
    )

    if output.questions:
        return {
            "retry_count": attempt,
            "status": "needs_clarification",
            "last_error": "\n".join(output.questions)
        }

    if not output.changes:
        return {
            "retry_count": attempt,
            "status": "no_changes",
            "last_error": "Developer generated no changes."
        }

    filesystem = FilesystemClient(
        workspace=state["workspace"]
    )

    await filesystem.connect()

    try:
        applied_files = await developer.apply_changes(
            output,
            filesystem
        )
    finally:
        await filesystem.disconnect()

    return {
        "retry_count": attempt,
        "implementation_summary": output.summary,
        "applied_files": applied_files,
        "status": "testing"
    }


async def tester_node(state: dict) -> dict:
    tester = TesterAgent()

    execution = await asyncio.to_thread(
        run_tests,
        state["workspace"]
    )

    test_output = (
        execution["stdout"] + "\n" + execution["stderr"]
    )

    analysis = await tester.analyze_results(
        task=state["request"],
        implementation_summary=state["implementation_summary"],
        test_output=test_output,
        exit_code=execution["exit_code"]
    )

    passed = (
        execution["exit_code"] == 0
        and not execution["timed_out"]
        and analysis.overall_status == "passed"
        and analysis.failed_count == 0
    )

    last_error = ""

    if not passed:
        last_error = f"""
        EXIT CODE: {execution['exit_code']}
        TIMED OUT: {execution['timed_out']}

        TEST OUTPUT:
        {test_output}

        TESTER ANALYSIS:
        {analysis.model_dump_json(indent=2)}
        """

    return {
        "execution_result": execution,
        "test_report": analysis.model_dump(),
        "status": "passed" if passed else "failed",
        "last_error": last_error
    }
