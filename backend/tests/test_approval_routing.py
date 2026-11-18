import asyncio
from uuid import uuid4

import pytest

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from backend.app.graph.state import CodeCrewState
from backend.app.graph.approval import human_approval_node
from backend.app.graph.routing import route_after_approval


async def check_decision(decision, initial_count, expected_status, expected_route):
    graph = StateGraph(CodeCrewState)

    graph.add_node("approval", human_approval_node)
    graph.add_edge(START, "approval")
    graph.add_edge("approval", END)

    app = graph.compile(checkpointer=InMemorySaver())

    config = {
        "configurable": {
            "thread_id": str(uuid4())
        }
    }

    initial_state = {
        "review_report": {"overall_status": "approved"},
        "implementation_summary": "Feature implemented.",
        "applied_files": ["calculator.py"],
        "test_report": {"overall_status": "passed"},
        "revision_count": initial_count,
        "max_revisions": 2
    }

    result = await app.ainvoke(initial_state, config=config)

    assert result.get("__interrupt__"), "Workflow did not pause."

    result = await app.ainvoke(
        Command(resume={
            "decision": decision,
            "feedback": "Test feedback"
        }),
        config=config
    )

    assert result["approval_status"] == expected_status
    assert result["revision_count"] == (
        initial_count + (1 if decision == "request_changes" else 0)
    )

    route = route_after_approval(result)
    assert route == expected_route

    print(f"{decision}: PASSED -> {route}")


@pytest.mark.parametrize(
    "decision, initial_count, expected_status, expected_route",
    [
        ("approve", 0, "approve", "approved"),
        ("request_changes", 0, "request_changes", "changes_requested"),
        ("reject", 0, "reject", "rejected"),
        ("request_changes", 2, "request_changes", "revision_limit_reached"),
    ]
)
async def test_approval_routing(
    decision, initial_count, expected_status, expected_route
):
    await check_decision(
        decision, initial_count, expected_status, expected_route
    )


async def main():
    await check_decision(
        "approve", 0, "approve", "approved"
    )

    await check_decision(
        "request_changes", 0, "request_changes", "changes_requested"
    )

    await check_decision(
        "reject", 0, "reject", "rejected"
    )

    await check_decision(
        "request_changes", 2, "request_changes", "revision_limit_reached"
    )

    print("\nAll approval routing tests passed!")


if __name__ == "__main__":
    asyncio.run(main())