import asyncio
import selectors
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from backend.app.graph.state import CodeCrewState
from backend.app.graph.approval import human_approval_node
from backend.app.database.checkpoint import pool, get_checkpointer


async def main():
    checkpointer = get_checkpointer()

    await pool.open()
    await checkpointer.setup()

    try:
        graph = StateGraph(CodeCrewState)

        graph.add_node("human_approval", human_approval_node)

        graph.add_edge(START, "human_approval")
        graph.add_edge("human_approval", END)

        app = graph.compile(checkpointer=checkpointer)

        config = {
            "configurable": {
                "thread_id": f"approval-test-{uuid4()}"
            }
        }

        initial_state = {
            "review_report": {
                "overall_status": "approved",
                "summary": "Implementation looks good."
            },
            "implementation_summary": "Added a new feature.",
            "applied_files": ["calculator.py"],
            "test_report": {
                "overall_status": "passed"
            }
        }

        print("\nStarting approval workflow...")

        result = await app.ainvoke(initial_state, config=config)

        print("\nWorkflow paused:")
        print(result.get("__interrupt__"))

        print("\nResuming with human approval...")

        result = await app.ainvoke(
            Command(
                resume={
                    "approved": True,
                    "feedback": "Looks good. Proceed."
                }
            ),
            config=config
        )

        print("\nFinal approval result:")
        print("Status:", result["approval_status"])
        print("Feedback:", result["approval_feedback"])

        assert result["approval_status"] == "approved"

        print("\nApproval interrupt/resume test passed!")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
    