
import asyncio
import shutil
from uuid import uuid4

import pytest

from backend.app.repository.repository_service import (
    clone_repository,
    validate_github_url
)
from backend.app.agents.repository_analyzer import (
    RepositoryAnalyzerAgent
)


REPO_URL = "https://github.com/pallets/flask"

# Clones a real repository and calls Groq.
pytestmark = pytest.mark.integration


async def test_repository_pipeline():
    run_id = str(uuid4())
    workspace = None

    try:
        print("\n1. Validating GitHub URL...")
        clone_url = validate_github_url(REPO_URL)
        print("PASSED:", clone_url)

        print("\n2. Cloning repository...")
        workspace = await clone_repository(
            clone_url,
            run_id
        )
        print("PASSED:", workspace)

        print("\n3. Running Repository Analyzer...")
        analyzer = RepositoryAnalyzerAgent()

        result = await analyzer.analyze_repository(
            workspace=workspace,
            request="Understand the existing application architecture."
        )

        print("\n4. Checking generated fields...")

        required_fields = [
            "repo_tree",
            "file_contents",
            "repository_analysis"
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            print("PASSED:", field)

        print("\n5. Checking repository analysis...")

        analysis = result["repository_analysis"]

        for field in [
            "repository_summary",
            "architecture_summary",
            "languages",
            "frameworks",
            "dependencies",
            "entry_points",
            "modules",
            "testing_setup",
            "important_files"
        ]:
            assert field in analysis, f"Missing analysis field: {field}"
            print("PASSED:", field)

        print("\nRepository pipeline completed successfully.")

    finally:
        if workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(test_repository_pipeline())
