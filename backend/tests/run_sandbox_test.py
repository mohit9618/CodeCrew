import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sandbox.execution import run_tests


if __name__ == "__main__":
    workspace = r"C:\CodeCrew\workspace"

    result = run_tests(workspace)

    print("\nEXIT CODE:", result["exit_code"])
    print("\nSTDOUT:\n", result["stdout"])
    print("\nSTDERR:\n", result["stderr"])
    print("\nTIMED OUT:", result["timed_out"])