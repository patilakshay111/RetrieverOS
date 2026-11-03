"""Managed PostgreSQL checkpoint lifecycle for LangGraph."""

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from sqlalchemy.engine import make_url


@dataclass
class PostgresCheckpointer:
    pool: ConnectionPool
    saver: PostgresSaver

    def close(self) -> None:
        self.pool.close()


def _psycopg_conninfo(database_url: str) -> str:
    url = make_url(database_url)
    if not url.drivername.startswith("postgresql"):
        raise ValueError("PostgreSQL checkpointer requires a PostgreSQL DATABASE_URL.")
    psycopg_url = url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    split = urlsplit(psycopg_url)
    query = [
        item
        for item in split.query.split("&")
        if item and not item.casefold().startswith("sslmode=allow")
    ]
    return urlunsplit((split.scheme, split.netloc, split.path, "&".join(query), ""))


def create_postgres_checkpointer(database_url: str) -> PostgresCheckpointer:
    conninfo = _psycopg_conninfo(database_url)
    pool = ConnectionPool(
        conninfo,
        min_size=1,
        max_size=10,
        timeout=5,
        kwargs={"autocommit": True},
        open=False,
    )
    initialized = False
    try:
        pool.open(wait=True, timeout=5)
        saver = PostgresSaver(pool)
        saver.setup()
        initialized = True
        return PostgresCheckpointer(pool=pool, saver=saver)
    finally:
        if not initialized:
            pool.close()
