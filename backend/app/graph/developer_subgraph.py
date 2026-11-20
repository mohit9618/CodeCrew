"""Development step: the developers carry out the architect's tasks.

Tasks run one after another in the order the architect planned
(`execution_batches`, which respects task dependencies). Each task goes to
the developer role it was assigned to; every role works through the same
change -> check -> fix loop (see agents/developer.py).
"""
from langgraph.graph import StateGraph, START, END

from backend.app.agents.backend_developer import BackendDeveloper
from backend.app.agents.database_developer import DatabaseDeveloper
from backend.app.agents.developer import DeveloperAgent
from backend.app.agents.frontend_developer import FrontendDeveloper
from backend.app.graph.state import CodeCrewState
from backend.app.mcp.clients.filesystem import FilesystemClient
from backend.app.utils.repo_tree import build_repo_tree
from backend.app.utils.retrieval import read_files_text
from backend.app.utils.text import clamp_text
from sandbox.execution import run_checks

# How much of the previous test failure the developer gets to see.
# The pytest summary is printed last, so the tail is what matters.
MAX_LAST_ERROR_CHARS = 1_500

# Characters of changed-file content kept in the state for the reviewer.
MAX_REVIEW_FILE_CHARS = 12_000

DEVELOPER_ROLES = {
    "backend_developer": BackendDeveloper,
    "database_developer": DatabaseDeveloper,
    "frontend_developer": FrontendDeveloper,
}


def order_tasks(tasks: list[dict], execution_batches: list[list[str]]) -> list[dict]:
    """Tasks in execution order.

    Follows `execution_batches`; tasks the batches do not mention (or all
    tasks, when there are no batches) keep their original order at the end.
    """
    by_id = {task["task_id"]: task for task in tasks}
    ordered = []

    for batch in execution_batches or []:
        for task_id in batch:
            task = by_id.pop(task_id, None)

            if task is not None:
                ordered.append(task)

    ordered.extend(by_id.values())

    return ordered


def build_task_text(task: dict, state: dict) -> str:
    """Task description sent to a developer agent.

    Includes the human review feedback and, on a retry, the previous test
    failure. Without the failure the developer would repeat the same
    mistake on every attempt. Empty sections are left out so they do not
    waste tokens.
    """
    parts = [str(task)]

    feedback = (state.get("approval_feedback") or "").strip()

    if feedback:
        parts.append(
            "HUMAN REVIEW FEEDBACK:\n"
            f"{feedback}\n\n"
            "Address this feedback while preserving existing functionality."
        )

    last_error = (state.get("last_error") or "").strip()

    if last_error:
        parts.append(
            "PREVIOUS TEST FAILURE (your last attempt did not pass):\n"
            + clamp_text(
                last_error,
                MAX_LAST_ERROR_CHARS,
                "previous test failure",
                keep="tail"
            )
            + "\n\nFix the implementation based on this failure. "
            "Preserve existing functionality and tests."
        )

    return "\n\n".join(parts)


def log_task(task: dict, role: str, task_text: str) -> None:
    print(
        f"\n========== {role.upper()}: TASK {task.get('task_id')} "
        f"- {task.get('title', '')} =========="
    )
    print(f"Task characters: {len(task_text):,}")


async def develop_node(state: CodeCrewState) -> dict:
    tasks = order_tasks(
        state.get("tasks", []), state.get("execution_batches", [])
    )

    if not tasks:
        return {
            "developer_stage": "skipped",
            "stage_summary": [],
        }

    workspace = state["workspace"]
    stack_summary = (state.get("stack") or {}).get("summary", "")

    # A retry after failed tests or review feedback may legitimately need
    # no further change for some tasks.
    is_rework = bool(
        state.get("last_error") or state.get("approval_feedback")
    )

    filesystem = FilesystemClient(workspace=workspace)
    await filesystem.connect()

    applied_files = list(state.get("applied_files", []))
    summaries = []
    agents: dict[str, DeveloperAgent] = {}

    try:
        for task in tasks:
            role = task["assigned_developer"]

            if role not in agents:
                agents[role] = DEVELOPER_ROLES.get(role, DeveloperAgent)()

            task_text = build_task_text(task, state)
            log_task(task, role, task_text)

            result = await agents[role].run_task(
                task,
                task_text=task_text,
                workspace=workspace,
                filesystem=filesystem,
                stack_summary=stack_summary,
                api_contract=state.get("api_contract", []),
                data_model=state.get("data_model", []),
                check_runner=run_checks,
                allow_no_changes=is_rework,
                applied_before=applied_files,
            )

            for path in result["applied_files"]:
                if path not in applied_files:
                    applied_files.append(path)

            if result["summary"]:
                summaries.append(result["summary"])

            for warning in result["warnings"]:
                print(f"[Developer] task {task['task_id']} warning: {warning}")

    finally:
        await filesystem.disconnect()

    return {
        "applied_files": applied_files,
        "working_repo_tree": build_repo_tree(workspace),
        "working_file_contents": read_files_text(
            workspace, applied_files, MAX_REVIEW_FILE_CHARS
        ),
        "stage_summary": summaries,
        "developer_stage": "development_completed",
    }


def initialize_developer_state(state: CodeCrewState) -> dict:
    print("DEVELOPER TASK ASSIGNMENTS:", [
        (task.get("task_id"), task.get("assigned_developer"))
        for task in state.get("tasks", [])
    ])

    return {
        "working_repo_tree": (
            state.get("working_repo_tree") or state.get("repo_tree", "")
        ),
        "working_file_contents": (
            state.get("working_file_contents") or state.get("file_contents", "")
        ),
        "applied_files": state.get("applied_files", []),
        "stage_summary": [],
        "developer_stage": "development",
    }


def finalize_development_node(state: CodeCrewState) -> dict:
    stage_summary = state.get("stage_summary") or []
    applied_files = state.get("applied_files") or []

    # Count this development pass. route_after_testing compares
    # retry_count with max_attempts; without this the counter stayed at 0
    # and a failing test run sent the workflow back to the developers
    # forever.
    attempt = state.get("retry_count", 0) + 1

    return {
        "developer_stage": "completed",
        "status": "testing",
        "retry_count": attempt,
        "implementation_summary": "\n".join(stage_summary),
        "applied_files": applied_files,
        "approval_feedback": ""
    }


graph = StateGraph(CodeCrewState)

graph.add_node("initialize", initialize_developer_state)
graph.add_node("develop", develop_node)
graph.add_node("finalize", finalize_development_node)

graph.add_edge(START, "initialize")
graph.add_edge("initialize", "develop")
graph.add_edge("develop", "finalize")
graph.add_edge("finalize", END)

developer_subgraph = graph.compile()
