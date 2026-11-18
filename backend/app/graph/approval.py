from langgraph.types import interrupt


def human_approval_node(state: dict) -> dict:
    approval_request = {
        "message": "Review the implementation and choose an action.",
        "options": ["approve", "request_changes", "reject"],
        "review_report": state.get("review_report", {}),
        "implementation_summary": state.get("implementation_summary", ""),
        "applied_files": state.get("applied_files", []),
        "test_report": state.get("test_report", {})
    }

    decision = interrupt(approval_request)

    if not isinstance(decision, dict):
        raise ValueError("Approval decision must be a dictionary.")

    action = decision.get("decision")

    if action not in ["approve", "request_changes", "reject"]:
        raise ValueError("Invalid approval decision.")

    feedback = decision.get("feedback", "")

    if not isinstance(feedback, str):
        raise ValueError("Feedback must be a string.")

    return {
        "approval_status": action,
        "approval_feedback": feedback,
        "revision_feedback": feedback if action == "request_changes" else "",
        "approval_request": approval_request,
        "revision_count": state.get("revision_count", 0) + (
            1 if action == "request_changes" else 0
        ),
        # A requested revision starts with a fresh retry budget and no
        # leftover failure from the previous round.
        "retry_count": (
            0 if action == "request_changes"
            else state.get("retry_count", 0)
        ),
        "last_error": (
            "" if action == "request_changes"
            else state.get("last_error", "")
        ),
    }
