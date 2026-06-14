"""tests/test_db.py
Unit tests for the database loader (load_database.py).
Verifies that the SQLite database has the correct tables, views, indexes,
row counts, and basic data integrity.
"""

import os
import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------
DB_PATH = Path(os.environ.get("ASSET_DB_PATH", "lseg_assets.db"))

EXPECTED_TABLES = {"machines", "software", "hotfixes", "vulnerabilities"}
EXPECTED_VIEWS = {
    "v_patch_compliance",
    "v_security_risk",
    "v_software_prevalence",
    "v_os_distribution",
    "v_dept_hardware",
    "v_dept_risk",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}. Run load_database.py first."
    connection = sqlite3.connect(str(DB_PATH))
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_db_exists():
    assert DB_PATH.exists(), f"Database file not found: {DB_PATH}"


def test_tables_exist(conn):
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for table in EXPECTED_TABLES:
        assert table in tables, f"Expected table `{table}` not found. Tables present: {tables}"


def test_views_exist(conn):
    views = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view'"
        ).fetchall()
    }
    for view in EXPECTED_VIEWS:
        assert view in views, f"Expected view `{view}` not found. Views present: {views}"


def test_indexes_exist(conn):
    indexes = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()[0]
    assert indexes >= 8, f"Expected at least 8 indexes, found {indexes}"


# ---------------------------------------------------------------------------
# Row count tests
# ---------------------------------------------------------------------------

def test_machines_row_count(conn):
    count = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
    assert count == 520, f"Expected 520 rows in machines, got {count}"


def test_software_minimum_rows(conn):
    count = conn.execute("SELECT COUNT(*) FROM software").fetchone()[0]
    assert count >= 5000, f"Expected at least 5,000 rows in software, got {count}"


def test_hotfixes_minimum_rows(conn):
    count = conn.execute("SELECT COUNT(*) FROM hotfixes").fetchone()[0]
    assert count >= 1000, f"Expected at least 1,000 rows in hotfixes, got {count}"


def test_vulnerabilities_not_empty(conn):
    count = conn.execute("SELECT COUNT(*) FROM vulnerabilities").fetchone()[0]
    assert count > 0, "vulnerabilities table is empty"


# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------

def test_no_null_pc_names_in_machines(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(machines)")]
    pc_col = next((c for c in cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    if pc_col:
        nulls = conn.execute(f"SELECT COUNT(*) FROM machines WHERE {pc_col} IS NULL").fetchone()[0]
        assert nulls == 0, f"Found {nulls} null PC names in machines table"


def test_software_all_have_pc_reference(conn):
    """Every software row should have a non-null PC name."""
    sw_cols = [r[1] for r in conn.execute("PRAGMA table_info(software)")]
    pc_col = next((c for c in sw_cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    if pc_col:
        nulls = conn.execute(f"SELECT COUNT(*) FROM software WHERE {pc_col} IS NULL").fetchone()[0]
        assert nulls == 0, f"Found {nulls} software rows with null PC name"


def test_machines_unique_pc_names(conn):
    cols = [r[1] for r in conn.execute("PRAGMA table_info(machines)")]
    pc_col = next((c for c in cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    if pc_col:
        total = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
        unique = conn.execute(f"SELECT COUNT(DISTINCT {pc_col}) FROM machines").fetchone()[0]
        assert total == unique, f"Duplicate PC names found: {total} rows but only {unique} unique names"


# ---------------------------------------------------------------------------
# View query tests
# ---------------------------------------------------------------------------

def test_views_are_queryable(conn):
    """All 6 views should return results without errors."""
    for view in EXPECTED_VIEWS:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            assert count >= 0  # just checking it runs without error
        except Exception as e:
            pytest.fail(f"View `{view}` failed to query: {e}")


# ---------------------------------------------------------------------------
# Category column test (Module 6)
# ---------------------------------------------------------------------------

def test_software_category_column_exists(conn):
    """category column should exist after running categorise_software.py."""
    sw_cols = [r[1] for r in conn.execute("PRAGMA table_info(software)")]
    assert "category" in sw_cols, (
        "No `category` column in software table. Run categorise_software.py first."
    )
