
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from backend.app.config import DATABASE_URL


pool = AsyncConnectionPool(
    conninfo=DATABASE_URL,
    min_size=1,
    max_size=10,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row
    },
    open=False
)


def get_checkpointer():
    return AsyncPostgresSaver(pool)
