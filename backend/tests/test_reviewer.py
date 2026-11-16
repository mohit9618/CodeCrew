
import asyncio
from backend.app.agents.reviewer import ReviewerAgent


async def main():
    reviewer = ReviewerAgent()

    result = await reviewer.review_implementation(
        request="Add cube(a) and tests for positive, negative and zero values.",
        project_spec={
            "acceptance_criteria": [
                "cube(a) returns a ** 3",
                "Tests cover positive, negative and zero values"
            ]
        },
        tasks=[
            {
                "task_id": "T1",
                "title": "Implement cube",
                "assigned_developer": "backend_developer"
            },
            {
                "task_id": "T2",
                "title": "Add cube tests",
                "assigned_developer": "backend_developer"
            }
        ],
        implementation_summary="Implemented cube(a) and added three tests.",
        applied_files=[
            "calculator.py",
            "tests/test_calculator.py"
        ],
        file_contents="""
def cube(a):
    return a ** 3


def test_cube_positive():
    assert cube(3) == 27


def test_cube_negative():
    assert cube(-2) == -8


def test_cube_zero():
    assert cube(0) == 0
""",
        test_report={
            "overall_status": "passed",
            "passed_count": 3,
            "failed_count": 0,
            "skipped_count": 0
        }
    )

    print(result.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
