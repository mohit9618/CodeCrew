
import asyncio
import selectors
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.config import DATABASE_URL
from backend.app.services.run_service import (
    run_workflow,
    get_workflow_state
)


class TestState(TypedDict):
    request: str
    status: str


async def test_node(state: TestState):
    return {"status": "completed"}


async def main():
    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row
        },
        open=False
    )

    await pool.open()

    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        graph = StateGraph(TestState)

        graph.add_node("test", test_node)
        graph.add_edge(START, "test")
        graph.add_edge("test", END)

        workflow = graph.compile(checkpointer=checkpointer)

        run_id = "codecrew-service-test-002"

        initial_state = {
            "request": "Test run service",
            "status": "running"
        }

        result = await run_workflow(
            workflow,
            run_id,
            initial_state
        )

        print("\nWorkflow result:")
        print(result)

        saved_state = await get_workflow_state(
            workflow,
            run_id
        )

        print("\nRetrieved workflow state:")
        print(saved_state)

        assert result["status"] == "completed"
        assert saved_state["run_id"] == run_id
        assert saved_state["values"]["status"] == "completed"

        print("\nRun service test passed!")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
