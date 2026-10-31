
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.database.checkpoint import pool, get_checkpointer
from backend.app.graph.workflow import compile_workflow
from backend.app.api.runs import router as runs_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()

    try:
        checkpointer = get_checkpointer()
        await checkpointer.setup()

        app.state.workflow = compile_workflow(checkpointer)

        print("PostgreSQL connected!")
        print("LangGraph workflow initialized!")

        yield

    finally:
        await pool.close()
        print("PostgreSQL connection closed!")


app = FastAPI(
    title="CodeCrew API",
    description="Multi-agent AI Software Engineering Team",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(runs_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to CodeCrew API",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }
