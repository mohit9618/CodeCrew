
import asyncio
import json

from backend.app.agents.architect import ArchitectAgent

project_spec = {
    "project_summary": "Add a power function to the calculator module.",
    "technology_domains": ["backend", "testing"],
    "functional_requirements": [
        "Implement power(base, exponent) in calculator.py.",
        "Return base raised to the exponent."
    ],
    "non_functional_requirements": [],
    "acceptance_criteria": [
        "The power function returns correct results.",
        "Existing tests continue to pass."
    ],
    "edge_cases": [
        "Positive exponent",
        "Zero exponent",
        "Negative exponent"
    ],
    "assumptions": [
        "The function should follow existing calculator conventions."
    ],
    "dependencies": [],
    "constraints": [
        "Do not break existing calculator functionality."
    ]
}

repo_tree = """
calculator.py
tests/
    test_calculator.py
"""

file_contents = """
File: calculator.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b


File: tests/test_calculator.py

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2
"""

async def main():
    agent = ArchitectAgent()

    result = await agent.design_architecture(
        project_spec=project_spec,
        repo_tree=repo_tree,
        file_contents=file_contents
    )

    print(json.dumps(result.model_dump(), indent=4))


if __name__ == "__main__":
    asyncio.run(main())
