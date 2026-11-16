from langgraph.graph import END
def route_after_development(state: dict) -> str:
    if state.get("status") == "testing":
        return "continue"

    return "end"


def route_after_testing(state: dict) -> str:
    if state.get("status") == "failed":
        if state.get("retry_count", 0) < state.get("max_attempts", 3):
            return "developer"
        return END

    if state.get("status") == "passed":
        return "reviewer"

    return END


def route_after_approval(state: dict) -> str:
    decision = state.get("approval_status")

    if decision == "approve":
        return "approved"

    if decision == "request_changes":
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_revisions", 2)

        if revision_count <= max_revisions:
            return "changes_requested"

        return "revision_limit_reached"

    if decision == "reject":
        return "rejected"

    return END