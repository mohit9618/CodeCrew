
import operator
from typing import Annotated, TypedDict


class CodeCrewState(TypedDict, total=False):
    # User request and repository information
    request: str
    repo_url: str
    workspace: str
    repo_tree: str
    file_contents: str

    # Planning information
    intake_result: dict
    project_spec: dict
    architecture: dict
    tasks: list[dict]
    execution_batches: list[list[str]]

    # Developer information
    implementation_summary: str
    applied_files: list[str]

    # Developer pipeline control
    developer_stage: str
    working_repo_tree: str
    working_file_contents: str
    stage_summary: Annotated[list[str], operator.add]

    # Parallel developer agents can append their patches
    developer_errors: Annotated[list[dict], operator.add]
    patches: Annotated[list[dict], operator.add]

    # Testing information
    execution_result: dict
    test_report: dict

    # Workflow control
    retry_count: int
    max_attempts: int
    status: str
    last_error: str

    # Human approval and publishing
    approval_status: str
    pr_url: str

    # Usage tracking
    cost_usd: float
