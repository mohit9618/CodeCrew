
def route_after_development(state: dict) -> str:
    if state.get("status") == "testing":
        return "continue"

    return "end"


def route_after_testing(state: dict) -> str:
    if state.get("status") != "failed":
        return "end"

    if state.get("retry_count", 0) < state.get("max_attempts", 3):
        return "retry"

    return "end"
