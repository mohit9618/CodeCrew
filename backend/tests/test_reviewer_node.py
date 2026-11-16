import asyncio

from backend.app.graph.nodes import reviewer_node


async def main():
    state = {
        "request": "Implement cube(a) and add tests.",
        "project_spec": {
            "acceptance_criteria": [
                "cube(a) returns a ** 3",
                "Tests cover positive, negative and zero inputs."
            ]
        },
        "tasks": [],
        "implementation_summary": "Implemented cube and added tests.",
        "applied_files": ["calculator.py", "tests/test_calculator.py"],
        "working_file_contents": """
def cube(a):
    return a ** 3
""",
        "file_contents": "",
        "test_report": {
            "overall_status": "passed",
            "passed_count": 3,
            "failed_count": 0,
            "results": []
        }
    }

    result = await reviewer_node(state)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())