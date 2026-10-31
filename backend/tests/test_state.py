
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.graph.state import CodeCrewState


def test_codecrew_state():
    state: CodeCrewState = {
        "request": "Add multiplication to calculator",
        "workspace": r"C:\CodeCrew\workspace",
        "retry_count": 0,
        "max_attempts": 3,
        "status": "running"
    }

    assert state["request"] == "Add multiplication to calculator"
    assert state["retry_count"] == 0
    assert state["max_attempts"] == 3

    print("CodeCrewState test passed!")


if __name__ == "__main__":
    test_codecrew_state()
