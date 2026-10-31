
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
from backend.app.database.checkpoint import pool, get_checkpointer


class TestState(TypedDict):
    message: str
    status: str


async def checkpoint_node(state: TestState):
    return {"status": "checkpoint_saved"}


def create_test_graph(checkpointer):
    graph = StateGraph(TestState)

    graph.add_node("checkpoint", checkpoint_node)
    graph.add_edge(START, "checkpoint")
    graph.add_edge("checkpoint", END)

    return graph.compile(checkpointer=checkpointer)


async def main():
    config = {
        "configurable": {
            "thread_id": "codecrew-resume-test"
        }
    }

    try:
        # First execution: save checkpoint.
        await pool.open()

        checkpointer = get_checkpointer()
        await checkpointer.setup()

        test_app = create_test_graph(checkpointer)

        initial_state = {
            "message": "CodeCrew recovery test",
            "status": "running"
        }

        result = await test_app.ainvoke(initial_state, config)

        print("\nFirst execution:")
        print(result)

        await pool.close()

        # Second execution: create a fresh connection and checkpointer.
        new_pool = AsyncConnectionPool(
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

        await new_pool.open()

        try:
            new_checkpointer = AsyncPostgresSaver(new_pool)
            recovered_app = create_test_graph(new_checkpointer)

            saved_state = await recovered_app.aget_state(config)

            print("\nRecovered checkpoint:")
            print(saved_state.values)

            assert saved_state.values["message"] == (
                "CodeCrew recovery test"
            )

            assert saved_state.values["status"] == "checkpoint_saved"

            print("\nCheckpoint recovery test passed!")

        finally:
            await new_pool.close()

    finally:
        if pool:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
