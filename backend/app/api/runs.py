
from uuid import uuid4

from fastapi import APIRouter, Request, HTTPException

from backend.app.schemas.requests import RunRequest
from backend.app.services.run_service import run_workflow , get_workflow_state , get_workflow_history



router = APIRouter(prefix="/runs", tags=["Runs"])


@router.post("/")
async def create_run(
    data: RunRequest,
    request: Request
):
    run_id = str(uuid4())

    initial_state = {
        "request": data.request,
        "repo_url": data.repo_url,
        "workspace": data.workspace,
        "repo_tree": data.repo_tree,
        "file_contents": data.file_contents,
        "retry_count": 0,
        "max_attempts": 3,
        "status": "running"
    }

    try:
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