"""Regression tests for the endless developer <-> tester retry loop."""
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from backend.app.graph import nodes
from backend.app.graph.approval import human_approval_node
from backend.app.graph.developer_subgraph import (
    MAX_LAST_ERROR_CHARS,
    build_task_text,
    finalize_development_node,
)
from backend.app.graph.routing import route_after_testing
from backend.app.graph.state import CodeCrewState


def build_outer_graph(tester_result, calls):
    """Same developer -> tester -> router wiring as workflow.py."""

    def developer(state):
        calls.append("developer")
        # Real finalize logic: this is where retry_count must go up.
        return finalize_development_node(
            {**state, "stage_summary": ["did work"], "applied_files": []}
        )

    def tester(state):
        return tester_result

    graph = StateGraph(CodeCrewState)
    graph.add_node("developer", developer)
    graph.add_node("tester", tester)
    graph.add_node("reviewer", lambda state: {"review_status": "approved"})

    graph.add_edge(START, "developer")
    graph.add_edge("developer", "tester")
    graph.add_conditional_edges(
        "tester",
        route_after_testing,
        {"developer": "developer", "reviewer": "reviewer", END: END}
    )
    graph.add_edge("reviewer", END)

    return graph.compile()


def test_failing_tests_stop_after_max_attempts():
    calls = []
    app = build_outer_graph(
        {"status": "failed", "last_error": "assert 1 == 2"},
        calls
    )

    # Default recursion limit (25). Before the fix the counter never moved
    # and this raised GraphRecursionError after 12 developer runs.
    try:
        result = app.invoke({"retry_count": 0, "max_attempts": 3})
    except GraphRecursionError:
        pytest.fail("developer <-> tester loop never terminated")

    assert calls.count("developer") == 3
    assert result["retry_count"] == 3
    assert result["status"] == "failed"


def test_passing_tests_go_to_reviewer_once():
    calls = []
    app = build_outer_graph({"status": "passed", "last_error": ""}, calls)

    result = app.invoke({"retry_count": 0, "max_attempts": 3})

    assert calls == ["developer"]
    assert result["review_status"] == "approved"


def test_finalize_counts_each_attempt():
    state = {"retry_count": 1, "stage_summary": ["a"], "applied_files": []}

    assert finalize_development_node(state)["retry_count"] == 2
    assert finalize_development_node({})["retry_count"] == 1


def test_developer_sees_previous_failure_on_retry():
    long_error = "noise " * 1000 + "FAILED test_x - AssertionError"

    text = build_task_text(
        {"task_id": "T1"},
        {"last_error": long_error, "approval_feedback": ""}
    )

    assert "PREVIOUS TEST FAILURE" in text
    assert "FAILED test_x - AssertionError" in text  # tail is kept
    assert len(text) < MAX_LAST_ERROR_CHARS + 600


def test_first_attempt_has_no_empty_feedback_sections():
    text = build_task_text({"task_id": "T1"}, {})

    assert "T1" in text
    assert "HUMAN REVIEW FEEDBACK" not in text
    assert "PREVIOUS TEST FAILURE" not in text


# ---------------------------------------------------------------- tester

class FakeTester:
    def __init__(self, analysis):
        self.analysis = analysis

    async def analyze_results(self, **kwargs):
        return self.analysis


def analysis(status, issues=()):
    return SimpleNamespace(
        overall_status=status,
        issues=list(issues),
        model_dump=lambda: {"overall_status": status},
    )


def make_tester_state():
    return {
        "workspace": "unused",
        "request": "do it",
        "implementation_summary": "done"
    }


def execution_result(passed, timed_out=False, output="", no_checks=False):
    checks = [] if no_checks else [{
        "project": ".",
        "name": "tests",
        "blocking": True,
        "ok": passed,
        "exit_code": 0 if passed else 1,
        "timed_out": timed_out,
        "output": output,
    }]

    return {
        "passed": passed,
        "exit_code": 0 if passed else 1,
        "timed_out": timed_out,
        "stdout": output,
        "stderr": "",
        "checks": checks,
        "no_checks": no_checks,
        "stack": "- .: python - flask",
    }


@pytest.mark.parametrize(
    "passed, timed_out, tester_status, expected",
    [
        # A green sandbox run must pass even if the LLM summary fell back
        # to needs_attention (parse error / oversized prompt).
        (True, False, "needs_attention", "passed"),
        (True, False, "passed", "passed"),
        (False, False, "passed", "failed"),
        (False, True, "passed", "failed"),
    ]
)
async def test_tester_node_trusts_the_sandbox_result(
    monkeypatch, passed, timed_out, tester_status, expected
):
    monkeypatch.setattr(
        nodes,
        "run_checks",
        lambda workspace, files: execution_result(
            passed, timed_out, output="FAILED test_x - assert 1 == 2"
        )
    )
    monkeypatch.setattr(
        nodes, "TesterAgent", lambda: FakeTester(analysis(tester_status))
    )

    result = await nodes.tester_node(make_tester_state())

    assert result["status"] == expected
    assert bool(result["last_error"]) == (expected == "failed")


async def test_failure_output_is_last_in_last_error(monkeypatch):
    """Only the tail of last_error reaches the developer, so the failing
    output must come after the tester's notes."""
    monkeypatch.setattr(
        nodes, "run_checks",
        lambda workspace, files: execution_result(
            False, output="FAILED test_x - assert 1 == 2"
        )
    )
    monkeypatch.setattr(
        nodes, "TesterAgent",
        lambda: FakeTester(analysis("failed", issues=["Fix the import"]))
    )

    result = await nodes.tester_node(make_tester_state())

    assert result["last_error"].index("Fix the import") < result[
        "last_error"
    ].index("FAILED test_x")


async def test_tester_node_skips_the_llm_when_nothing_was_run(monkeypatch):
    monkeypatch.setattr(
        nodes, "run_checks",
        lambda workspace, files: execution_result(
            True, output="No runnable project", no_checks=True
        )
    )

    def must_not_be_created():
        raise AssertionError("no LLM call is needed when nothing ran")

    monkeypatch.setattr(nodes, "TesterAgent", must_not_be_created)

    result = await nodes.tester_node(make_tester_state())

    assert result["status"] == "passed"
    assert result["test_report"]["overall_status"] == "needs_attention"
    assert "No runnable project" in result["test_report"]["analysis"]


# ------------------------------------------------------------- publisher

async def test_publisher_is_skipped_without_a_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = await nodes.publisher_node({"request": "x", "workspace": "w"})

    assert result["publish_status"] == "skipped"
    assert "GITHUB_TOKEN" in result["publish_error"]
    assert result["status"] == "approved"


async def test_publisher_records_the_pull_request(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    seen = {}

    def fake_publish(workspace, files, title, body, branch, token):
        seen.update(files=files, branch=branch, token=token, title=title)
        return {"branch": branch, "pr_url": "https://github.com/o/r/pull/1"}

    monkeypatch.setattr(nodes, "publish_changes", fake_publish)

    result = await nodes.publisher_node({
        "request": "Add a health endpoint",
        "workspace": "/ws/abcd1234-0000",
        "applied_files": ["app/health.py"],
    })

    assert result["status"] == "published"
    assert result["pr_url"] == "https://github.com/o/r/pull/1"
    assert seen["files"] == ["app/health.py"]
    assert seen["branch"].startswith("codecrew/add-a-health-endpoint-")


@pytest.mark.parametrize("error", [
    nodes.PublishError("push refused"),
    RuntimeError("boom"),
])
async def test_publisher_never_raises_into_the_workflow(monkeypatch, error):
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")

    def failing(*args):
        raise error

    monkeypatch.setattr(nodes, "publish_changes", failing)

    result = await nodes.publisher_node({"request": "x", "workspace": "/ws/a"})

    assert result["publish_status"] == "failed"
    assert str(error) in result["publish_error"]
    assert result["status"] == "approved"


# -------------------------------------------------------------- approval

async def test_requested_revision_resets_retry_budget():
    graph = StateGraph(CodeCrewState)
    graph.add_node("approval", human_approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)

    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}

    await app.ainvoke(
        {"retry_count": 2, "last_error": "old failure", "revision_count": 0},
        config=config
    )
    result = await app.ainvoke(
        Command(resume={"decision": "request_changes", "feedback": "x"}),
        config=config
    )

    assert result["retry_count"] == 0
    assert result["last_error"] == ""
    assert result["revision_count"] == 1


# ------------------------------------------------------------- small units

def test_sandbox_timeout_output_is_text():
    from sandbox.execution import _as_text

    assert _as_text(None) == ""
    assert _as_text(b"caf\xc3\xa9") == "café"
    assert _as_text("already text") == "already text"


def test_filesystem_client_command_is_platform_aware(monkeypatch):
    from backend.app.mcp.clients.filesystem import FilesystemClient

    monkeypatch.setattr(sys, "platform", "linux")
    client = FilesystemClient(workspace="/tmp/ws")
    assert client.server_params.command == "npx"

    monkeypatch.setattr(sys, "platform", "win32")
    client = FilesystemClient(workspace="C:\\ws")
    assert client.server_params.command == "cmd"
    assert client.server_params.args[:2] == ["/c", "npx"]
