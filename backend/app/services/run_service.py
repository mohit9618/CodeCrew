
from langgraph.graph.state import CompiledStateGraph


async def run_workflow(
    workflow: CompiledStateGraph,
    run_id: str,
    initial_state: dict
):
    config = {
        "configurable": {
            "thread_id": run_id
        }
    }

    result = await workflow.ainvoke(
        initial_state,
        config
    )

    return result


async def get_workflow_state(
    workflow: CompiledStateGraph,
    run_id: str
):
    config = {
        "configurable": {
            "thread_id": run_id
        }
    }

    state = await workflow.aget_state(config)

    if not state.values:
        return None

    return {
        "run_id": run_id,
        "values": state.values,
        "next": list(state.next),
        "metadata": state.metadata
    }


async def get_workflow_history(
    workflow: CompiledStateGraph,
    run_id: str
):
    config = {
        "configurable": {
            "thread_id": run_id
        }
    }

    history = []

    async for state in workflow.aget_state_history(config):
        history.append({
            "values": state.values,
            "next": list(state.next),
            "metadata": state.metadata
        })

    return {
        "run_id": run_id,
        "total_checkpoints": len(history),
        "history": history
    }