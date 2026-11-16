
from pathlib import Path, PurePosixPath

from langgraph.graph import StateGraph, START, END

from backend.app.graph.state import CodeCrewState
from backend.app.agents.backend_developer import BackendDeveloper
from backend.app.agents.database_developer import DatabaseDeveloper
from backend.app.agents.frontend_developer import FrontendDeveloper
from backend.app.mcp.clients.filesystem import FilesystemClient


def get_safe_paths(paths: list[str], workspace: str) -> list[str]:
    safe_paths = []
    root = Path(workspace).resolve()

    for path in paths:
        rel_path = PurePosixPath(path.replace("\\", "/"))

        if (
            rel_path.is_absolute()
            or ".." in rel_path.parts
            or not rel_path.parts
            or ":" in path
        ):
            raise ValueError(f"Unsafe repository path: {path}")

        full_path = (root / Path(*rel_path.parts)).resolve()
        full_path.relative_to(root)

        if str(full_path) not in safe_paths:
            safe_paths.append(str(full_path))

    return safe_paths


async def backend_stage_node(state: CodeCrewState) -> dict:
    tasks = [
        task for task in state.get("tasks", [])
        if task["assigned_developer"] == "backend_developer"
    ]

    if not tasks:
        return {
            "developer_stage": "backend_skipped",
            "status": "backend_skipped"
        }

    filesystem = FilesystemClient(workspace=state["workspace"])
    await filesystem.connect()

    applied_files = list(state.get("applied_files", []))
    summaries = []

    try:
        developer = BackendDeveloper()

        for task in tasks:
            output = await developer.generate_changes(
                task=f"""
                {task}

                HUMAN REVIEW FEEDBACK:
                {state.get("approval_feedback", "")}

                Address this feedback while preserving existing functionality.
                """,
                repo_tree=state.get("working_repo_tree", state.get("repo_tree", "")),
                file_contents=state.get("working_file_contents", state.get("file_contents", ""))
            )

            if output.questions:
                raise RuntimeError(
                    f"Backend task {task['task_id']} has blockers: {output.questions}"
                )

            if not output.changes:
                raise RuntimeError(
                    f"Backend task {task['task_id']} generated no changes."
                )

            files = await developer.apply_changes(output, filesystem)

            applied_files.extend(files)
            summaries.append(output.summary)

            read_paths = get_safe_paths(
                list(dict.fromkeys(
                    applied_files
                    + task.get("files_to_read", [])
                    + task.get("files_to_modify", [])
                )),
                state["workspace"]
            )

            updated_contents = (
                await filesystem.read_multiple_files(read_paths)
                if read_paths else state.get("working_file_contents", state.get("file_contents", ""))
            )

            updated_tree = await filesystem.get_directory_tree()

            state = {
                **state,
                "working_file_contents": updated_contents,
                "working_repo_tree": updated_tree
            }

        return {
            "applied_files": list(dict.fromkeys(applied_files)),
            "working_repo_tree": updated_tree,
            "working_file_contents": updated_contents,
            "stage_summary": state.get("stage_summary", []) + summaries,
            "developer_stage": "backend_completed",
            "status": "backend_completed"
        }

    finally:
        await filesystem.disconnect()


async def database_stage_node(state: CodeCrewState) -> dict:
    tasks = [
        task for task in state.get("tasks", [])
        if task["assigned_developer"] == "database_developer"
    ]

    if not tasks:
        return {
            "developer_stage": "database_skipped",
            "status": "database_skipped"
        }

    filesystem = FilesystemClient(workspace=state["workspace"])
    await filesystem.connect()

    applied_files = list(state.get("applied_files", []))
    summaries = []

    try:
        developer = DatabaseDeveloper()

        for task in tasks:
            output = await developer.generate_changes(
                task=f"""
                {task}

                HUMAN REVIEW FEEDBACK:
                {state.get("approval_feedback", "")}

                Address this feedback while preserving existing functionality.
                """,
                repo_tree=state.get("working_repo_tree", ""),
                file_contents=state.get("working_file_contents", "")
            )

            if output.questions:
                raise RuntimeError(
                    f"Database task {task['task_id']} has blockers: {output.questions}"
                )

            if not output.changes:
                raise RuntimeError(
                    f"Database task {task['task_id']} generated no changes."
                )

            files = await developer.apply_changes(output, filesystem)

            applied_files.extend(files)
            summaries.append(output.summary)

            read_paths = get_safe_paths(
                list(dict.fromkeys(
                    applied_files
                    + task.get("files_to_read", [])
                    + task.get("files_to_modify", [])
                )),
                state["workspace"]
            )

            updated_contents = (
                await filesystem.read_multiple_files(read_paths)
                if read_paths else state.get("working_file_contents", "")
            )

            updated_tree = await filesystem.get_directory_tree()

            state = {
                **state,
                "working_file_contents": updated_contents,
                "working_repo_tree": updated_tree
            }

        return {
            "applied_files": list(dict.fromkeys(applied_files)),
            "working_repo_tree": updated_tree,
            "working_file_contents": updated_contents,
            "stage_summary": state.get("stage_summary", []) + summaries,
            "developer_stage": "database_completed",
            "status": "database_completed"
        }

    finally:
        await filesystem.disconnect()


async def frontend_stage_node(state: CodeCrewState) -> dict:
    tasks = [
        task for task in state.get("tasks", [])
        if task["assigned_developer"] == "frontend_developer"
    ]

    if not tasks:
        return {
            "developer_stage": "frontend_skipped",
            "status": "frontend_skipped"
        }

    filesystem = FilesystemClient(workspace=state["workspace"])
    await filesystem.connect()

    applied_files = list(state.get("applied_files", []))
    summaries = []

    try:
        developer = FrontendDeveloper()

        for task in tasks:
            output = await developer.generate_changes(
                task=f"""
                {task}

                HUMAN REVIEW FEEDBACK:
                {state.get("approval_feedback", "")}

                Address this feedback while preserving existing functionality.
                """,
                repo_tree=state.get("working_repo_tree", ""),
                file_contents=state.get("working_file_contents", "")
            )

            if output.questions:
                raise RuntimeError(
                    f"Frontend task {task['task_id']} has blockers: {output.questions}"
                )

            if not output.changes:
                raise RuntimeError(
                    f"Frontend task {task['task_id']} generated no changes."
                )

            files = await developer.apply_changes(output, filesystem)

            applied_files.extend(files)
            summaries.append(output.summary)

            read_paths = get_safe_paths(
                list(dict.fromkeys(
                    applied_files
                    + task.get("files_to_read", [])
                    + task.get("files_to_modify", [])
                )),
                state["workspace"]
            )

            updated_contents = (
                await filesystem.read_multiple_files(read_paths)
                if read_paths else state.get("working_file_contents", "")
            )

            updated_tree = await filesystem.get_directory_tree()

            state = {
                **state,
                "working_file_contents": updated_contents,
                "working_repo_tree": updated_tree
            }

        return {
            "applied_files": list(dict.fromkeys(applied_files)),
            "working_repo_tree": updated_tree,
            "working_file_contents": updated_contents,
            "stage_summary": state.get("stage_summary", []) + summaries,
            "developer_stage": "frontend_completed",
            "status": "frontend_completed"
        }

    finally:
        await filesystem.disconnect()


def initialize_developer_state(state: CodeCrewState) -> dict:
    print("DEVELOPER TASK ASSIGNMENTS:", [
    (task.get("task_id"), task.get("assigned_developer"))
    for task in state.get("tasks", [])
    ])
    return {
        "working_repo_tree": state.get("working_repo_tree", state.get("repo_tree", "")),
        "working_file_contents": state.get("working_file_contents", state.get("file_contents", "")),
        "applied_files": state.get("applied_files", []),
        "stage_summary": [],
        "developer_stage": "backend"
    }

def finalize_development_node(state: CodeCrewState) -> dict:
    stage_summary = state.get("stage_summary") or []
    applied_files = state.get("applied_files") or []

    return {
        "developer_stage": "completed",
        "status": "testing",
        "implementation_summary": "\n".join(stage_summary),
        "applied_files": applied_files,
        "approval_feedback": "",
    }

graph = StateGraph(CodeCrewState)

graph.add_node("initialize", initialize_developer_state)
graph.add_node("backend", backend_stage_node)
graph.add_node("database", database_stage_node)
graph.add_node("frontend", frontend_stage_node)
graph.add_node("finalize", finalize_development_node)

graph.add_edge(START, "initialize")
graph.add_edge("initialize", "backend")
graph.add_edge("backend", "database")
graph.add_edge("database", "frontend")
graph.add_edge("frontend", "finalize")
graph.add_edge("finalize", END)

developer_subgraph = graph.compile()


