
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from backend.app.config import DATABASE_URL


pool = ConnectionPool(
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

checkpointer = PostgresSaver(pool)
