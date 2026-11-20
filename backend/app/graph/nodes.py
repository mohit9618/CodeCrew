import asyncio
import os

from backend.app.agents.intake import IntakeAgent
from backend.app.agents.project_manager import ProjectManagerAgent
from backend.app.agents.architect import ArchitectAgent
from backend.app.agents.tester import TesterAgent
from backend.app.agents.reviewer import ReviewerAgent
from backend.app.services.publisher import (
    PublishError,
    build_branch_name,
    build_pull_request_body,
    publish_changes,
)
from sandbox.execution import run_checks


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
        intake_result=state["intake_result"],
        repository_analysis=state["repository_analysis"]
    )

    return {
        "project_spec": output.model_dump(),
        "status": "specification_completed"
    }


async def architect_node(state: dict) -> dict:
    architect = ArchitectAgent()

    output = await architect.design_architecture(
        project_spec=state["project_spec"],
        repo_tree=state.get("working_repo_tree") or state["repo_tree"],
        file_contents=state.get("working_file_contents") or state["file_contents"],
        repository_analysis=state["repository_analysis"],
        revision_feedback=state.get("revision_feedback", "")
    )

    architecture = output.model_dump()

    return {
        "architecture": architecture,
        "tasks": architecture["tasks"],
        "execution_batches": architecture["execution_batches"],
        "api_contract": architecture.get("api_contract", []),
        "data_model": architecture.get("data_model", []),
        "status": "architecture_completed"
    }



async def tester_node(state: dict) -> dict:
    execution = await asyncio.to_thread(
        run_checks,
        state["workspace"],
        state.get("applied_files", [])
    )

    # The sandbox result is the source of truth. The LLM analysis below is
    # only a readable summary: if it fails to parse or is cut short it must
    # not turn a green run into a failure and restart the retry loop.
    passed = execution["passed"]

    print(
        f"\n[Tester] {'PASSED' if passed else 'FAILED'} "
        f"(exit_code={execution['exit_code']}, "
        f"timed_out={execution['timed_out']})"
    )

    for check in execution["checks"]:
        status = (
            "ok" if check["ok"]
            else "FAILED" if check["blocking"]
            else "warning"
        )
        print(f"  [{check['project']}] {check['name']}: {status}")

    if execution["no_checks"]:
        # Nothing was run, so there is nothing for the LLM to analyse.
        test_report = {
            "overall_status": "needs_attention",
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "results": [],
            "analysis": execution["stdout"],
            "issues": [],
        }
        issues = []
    else:
        tester = TesterAgent()

        analysis = await tester.analyze_results(
            task=state["request"],
            implementation_summary=state["implementation_summary"],
            test_output=execution["stdout"],
            exit_code=execution["exit_code"]
        )

        test_report = analysis.model_dump()
        issues = analysis.issues

    last_error = ""

    if not passed:
        failed_output = "\n\n".join(
            f"[{check['project']}] {check['name']}"
            + (" (timed out)" if check["timed_out"] else "")
            + f" exit {check['exit_code']}:\n{check['output'].strip()}"
            for check in execution["checks"]
            if check["blocking"] and not check["ok"]
        )

        issue_text = "\n".join(f"- {issue}" for issue in issues)

        # The failing output goes last: only its tail reaches the
        # developers' prompt.
        last_error = (
            (f"TESTER NOTES:\n{issue_text}\n\n" if issue_text else "")
            + f"FAILED CHECKS:\n{failed_output}"
        )

    return {
        "execution_result": {
            key: execution[key]
            for key in ("passed", "exit_code", "timed_out", "stack", "checks")
        },
        "test_report": test_report,
        "status": "passed" if passed else "failed",
        "last_error": last_error
    }


async def reviewer_node(state: dict) -> dict:
    reviewer = ReviewerAgent()

    result = await reviewer.review_implementation(
        request=state["request"],
        project_spec=state["project_spec"],
        tasks=state["tasks"],
        implementation_summary=state["implementation_summary"],
        applied_files=state["applied_files"],
        file_contents=state.get(
            "working_file_contents",
            state["file_contents"]
        ),
        test_report=state["test_report"]
    )

    review_report = result.model_dump()

    return {
        "review_report": review_report,
        "review_status": result.overall_status
    }


async def publisher_node(state: dict) -> dict:
    """After approval: commit, push and open a pull request.

    Never raises. If publishing is not possible the changes stay in the
    workspace and the reason is recorded in the state.
    """
    token = os.getenv("GITHUB_TOKEN", "")

    if not token:
        return {
            "status": "approved",
            "publish_status": "skipped",
            "publish_error": (
                "GITHUB_TOKEN is not set, so no pull request was opened. "
                "The changes remain in the run's workspace."
            ),
        }

    request = state.get("request", "")
    title = f"CodeCrew: {' '.join(request.split())[:70]}"
    branch = build_branch_name(request, state["workspace"])

    try:
        result = await asyncio.to_thread(
            publish_changes,
            state["workspace"],
            state.get("applied_files", []),
            title,
            build_pull_request_body(state),
            branch,
            token,
        )

    except PublishError as error:
        return {
            "status": "approved",
            "publish_status": "failed",
            "publish_error": str(error),
            "branch_name": branch,
        }

    except Exception as error:
        return {
            "status": "approved",
            "publish_status": "failed",
            "publish_error": f"{type(error).__name__}: {error}",
            "branch_name": branch,
        }

    print(f"[Publisher] Pull request opened: {result['pr_url']}")

    return {
        "status": "published",
        "publish_status": "published",
        "publish_error": "",
        "branch_name": result["branch"],
        "publish_mode": result["mode"],
        "pr_url": result["pr_url"],
    }
