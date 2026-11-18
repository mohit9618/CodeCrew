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


def analysis(status, failed=0):
    return SimpleNamespace(
        overall_status=status,
        failed_count=failed,
        model_dump=lambda: {"overall_status": status},
        model_dump_json=lambda indent=2: "{}"
    )


def make_tester_state():
    return {
        "workspace": "unused",
        "request": "do it",
        "implementation_summary": "done"
    }


@pytest.mark.parametrize(
    "exit_code, timed_out, tester_status, failed_count, expected",
    [
        # Green pytest run must pass even if the LLM summary fell back to
        # needs_attention (parse error / oversized prompt).
        (0, False, "needs_attention", 0, "passed"),
        (0, False, "passed", 0, "passed"),
        (1, False, "passed", 0, "failed"),
        (0, True, "passed", 0, "failed"),
        (0, False, "passed", 2, "failed"),
    ]
)
async def test_tester_node_trusts_pytest_exit_code(
    monkeypatch, exit_code, timed_out, tester_status, failed_count, expected
):
    monkeypatch.setattr(
        nodes,
        "run_tests",
        lambda workspace: {
            "exit_code": exit_code,
            "stdout": "out",
            "stderr": "",
            "timed_out": timed_out
        }
    )
    monkeypatch.setattr(
        nodes,
        "TesterAgent",
        lambda: FakeTester(analysis(tester_status, failed_count))
    )

    result = await nodes.tester_node(make_tester_state())

    assert result["status"] == expected
    assert bool(result["last_error"]) == (expected == "failed")


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
