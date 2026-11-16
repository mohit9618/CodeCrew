
import asyncio

from backend.app.graph import nodes


class MockArchitecture:
    def model_dump(self):
        return {
            "architecture_summary": "Revision planned",
            "tasks": [
                {
                    "task_id": "R1",
                    "title": "Fix factorial validation",
                    "description": "Reject boolean values.",
                    "domain": "backend",
                    "assigned_developer": "backend_developer",
                    "files_to_modify": ["calculator.py"],
                    "files_to_read": [],
                    "dependencies": [],
                    "acceptance_criteria": ["True raises TypeError"]
                }
            ],
            "execution_batches": [["R1"]]
        }


class MockArchitect:
    def __init__(self):
        self.received = {}

    async def design_architecture(self, **kwargs):
        self.received = kwargs
        return MockArchitecture()


async def main():
    mock_architect = MockArchitect()
    nodes.ArchitectAgent = lambda: mock_architect

    state = {
        "project_spec": {"goal": "Implement factorial"},
        "repo_tree": "old_tree",
        "file_contents": "old_code",
        "working_repo_tree": "updated_tree",
        "working_file_contents": "updated_code",
        "revision_feedback": "Reject boolean values in factorial."
    }

    result = await nodes.architect_node(state)

    received = mock_architect.received

    assert received["project_spec"] == state["project_spec"]
    assert received["repo_tree"] == "updated_tree"
    assert received["file_contents"] == "updated_code"
    assert received["revision_feedback"] == state["revision_feedback"]

    assert result["tasks"][0]["task_id"] == "R1"
    assert result["execution_batches"] == [["R1"]]

    print("Project specification: PASSED")
    print("Latest repository tree: PASSED")
    print("Latest repository contents: PASSED")
    print("Revision feedback: PASSED")
    print("Revision tasks generated: PASSED")
    print("\nAll Architect revision tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
