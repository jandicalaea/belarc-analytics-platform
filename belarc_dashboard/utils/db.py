"""
Database access layer for the Enterprise IT Asset Intelligence Platform dashboard.

All Streamlit pages import from this module. It centralises:
  - locating the SQLite database (lseg_assets.db)
  - opening read-only connections
  - cached query helpers (so re-renders don't re-hit the DB)
  - schema introspection (tables / views / columns)
  - small resilience helpers for working with views whose exact
    column names may vary between runs of load_database.py
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Locating the database
# ---------------------------------------------------------------------------
# Resolution order:
#   1. LSEG_DB_PATH environment variable (useful for AWS / CI / Docker)
#   2. lseg_assets.db sitting next to this project
#   3. The author's local Downloads path (Windows dev machine)
#   4. lseg_assets.db in the current working directory
_DEFAULT_CANDIDATES = [
    os.environ.get("LSEG_DB_PATH", ""),
    str(Path(__file__).resolve().parents[1] / "lseg_assets.db"),
    str(Path(__file__).resolve().parents[2] / "lseg_assets.db"),
    r"C:\Users\oli\Downloads\lseg_assets.db",
    "lseg_assets.db",
]


def resolve_db_path() -> str | None:
    """Return the first database path that exists, or None if none are found."""
    for candidate in _DEFAULT_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def get_connection() -> sqlite3.Connection:
    """
    Open a read-only SQLite connection.

    Using the file: URI with mode=ro guarantees the dashboard can never
    mutate the asset database, which is the correct posture for a BI tool.
    """
    db_path = resolve_db_path()
    if db_path is None:
        st.error(
            "Could not locate **lseg_assets.db**.\n\n"
            "Set the `LSEG_DB_PATH` environment variable, or place the database "
            "next to the dashboard. Run `load_database.py` first if you have not."
        )
        st.stop()

    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


# ---------------------------------------------------------------------------
# Cached query helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SELECT and return a DataFrame. Cached for 10 minutes."""
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=600, show_spinner=False)
def load_view(name: str) -> pd.DataFrame:
    """Read an entire view/table by name (used for the pre-built analytical views)."""
    return run_query(f"SELECT * FROM {name}")


@st.cache_data(ttl=600, show_spinner=False)
def list_objects() -> pd.DataFrame:
    """Return all tables and views in the database with their type."""
    return run_query(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY type, name"
    )


@st.cache_data(ttl=600, show_spinner=False)
def columns_of(name: str) -> list[str]:
    """Return the column names for a given table or view."""
    info = run_query(f"PRAGMA table_info({name})")
    if info.empty:
        return []
    return info["name"].tolist()


def object_exists(name: str) -> bool:
    objs = list_objects()
    return not objs.empty and name in set(objs["name"])


# ---------------------------------------------------------------------------
# Resilience helpers
# ---------------------------------------------------------------------------
def first_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """
    Return the first column in `candidates` that exists in `df`.

    The analytical views were authored by hand, so naming may drift slightly
    (e.g. risk_score vs security_score). This lets charts adapt instead of
    crashing on a KeyError.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes("number").columns.tolist()
