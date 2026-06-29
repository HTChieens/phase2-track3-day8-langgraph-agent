"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _sqlite_path(database_url: str | None) -> Path:
    """Convert a SQLite URL or plain path into a filesystem path."""
    if not database_url:
        return Path("outputs/checkpoints.sqlite")
    if database_url.startswith("sqlite:///"):
        return Path(database_url.removeprefix("sqlite:///"))
    if database_url.startswith("sqlite://"):
        return Path(database_url.removeprefix("sqlite://"))
    return Path(database_url)


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    The default backend is MemorySaver. SQLite is available for durable checkpoint evidence.
    Postgres remains an optional extension task.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    normalized_kind = kind.lower().strip()

    if normalized_kind == "none":
        return None
    if normalized_kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if normalized_kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "Install SQLite checkpoint support with: "
                "pip install langgraph-checkpoint-sqlite"
            ) from exc

        path = _sqlite_path(database_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        saver = SqliteSaver(conn=conn)
        setup = getattr(saver, "setup", None)
        if callable(setup):
            setup()
        return saver

    if normalized_kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension and is not implemented"
        )

    raise ValueError(f"Unknown checkpointer kind: {kind}")
