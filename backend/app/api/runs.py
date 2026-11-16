
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException

from backend.app.schemas.requests import RunRequest, ApprovalRequest

from backend.app.services.run_service import (
    run_workflow,
    get_workflow_state,
    get_workflow_history,
    resume_workflow
)

from backend.app.repository.repository_service import (
    clone_repository,
    RepositoryError
)

from backend.app.agents.repository_analyzer import (
    RepositoryAnalyzerAgent
)


router = APIRouter(prefix="/runs", tags=["Runs"])


@router.post("/")
async def create_run(
    data: RunRequest,
    request: Request
):
    run_id = str(uuid4())

    try:
        workspace = await clone_repository(
            str(data.repo_url),
            run_id
        )

        analyzer = RepositoryAnalyzerAgent()

        analysis = await analyzer.analyze_repository(
            workspace=workspace,
            request=data.request
        )

        initial_state = {
            "request": data.request,
            "repo_url": str(data.repo_url),
            "workspace": workspace,
            "repo_tree": analysis["repo_tree"],
            "file_contents": analysis["file_contents"],
            "repository_analysis": analysis["repository_analysis"],
            "retry_count": 0,
            "max_attempts": 3,
            "revision_count": 0,
            "max_revisions": 2,
            "status": "running"
        }

        result = await run_workflow(
            request.app.state.workflow,
            run_id,
            initial_state
        )

        return {
            "run_id": run_id,
            "status": result.get("status"),
            "result": result
        }

    except RepositoryError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/{run_id}")
async def get_run_status(
    run_id: str,
    request: Request
):
    try:
        result = await get_workflow_state(
            request.app.state.workflow,
            run_id
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Run not found"
            )

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/{run_id}/history")
async def get_run_history(
    run_id: str,
    request: Request
):
    try:
        result = await get_workflow_history(
            request.app.state.workflow,
            run_id
        )

        if result["total_checkpoints"] == 0:
            raise HTTPException(
                status_code=404,
                detail="Run history not found"
            )

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.post("/{run_id}/resume")
async def resume_run(
    run_id: str,
    data: ApprovalRequest,
    request: Request
):
    try:
        result = await resume_workflow(
            request.app.state.workflow,
            run_id,
            data.decision,
            data.feedback
        )

        return {
            "run_id": run_id,
            "status": result.get("status"),
            "result": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
