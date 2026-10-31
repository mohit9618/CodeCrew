
from langgraph.graph import StateGraph, START, END

from backend.app.graph.state import CodeCrewState
from backend.app.graph.nodes import (
    developer_node,
    tester_node
)
from backend.app.graph.routing import (
    route_after_development,
    route_after_testing
)


graph = StateGraph(CodeCrewState)

graph.add_node("developer", developer_node)
graph.add_node("tester", tester_node)

graph.add_edge(START, "developer")

graph.add_conditional_edges(
    "developer",
    route_after_development,
    {
        "continue": "tester",
        "end": END
    }
)

graph.add_conditional_edges(
    "tester",
    route_after_testing,
    {
        "retry": "developer",
        "end": END
    }
)


def compile_workflow(checkpointer):
    return graph.compile(checkpointer=checkpointer)
