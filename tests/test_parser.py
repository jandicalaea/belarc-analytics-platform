"""tests/test_parser.py
Unit tests for the ETL parser (parse_belarc.py).
Verifies that parsed CSVs exist, have the expected columns, and contain data.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths — CI sets ASSET_DB_PATH; locally falls back to relative paths
# ---------------------------------------------------------------------------
PARSED_DIR = Path(os.environ.get("PARSED_OUTPUT_DIR", "parsed_output"))


def _csv(name: str) -> Path:
    return PARSED_DIR / name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def machines_df():
    path = _csv("machines.csv")
    assert path.exists(), f"machines.csv not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def software_df():
    path = _csv("software.csv")
    assert path.exists(), f"software.csv not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def hotfixes_df():
    path = _csv("hotfixes.csv")
    assert path.exists(), f"hotfixes.csv not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def vulnerabilities_df():
    path = _csv("vulnerabilities.csv")
    assert path.exists(), f"vulnerabilities.csv not found at {path}"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# machines.csv tests
# ---------------------------------------------------------------------------

def test_machines_not_empty(machines_df):
    assert len(machines_df) > 0, "machines.csv is empty"


def test_machines_row_count(machines_df):
    """Should have exactly 520 rows — one per generated PC."""
    assert len(machines_df) == 520, f"Expected 520 rows, got {len(machines_df)}"


def test_machines_has_pc_name_column(machines_df):
    pc_cols = [c for c in machines_df.columns if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")]
    assert len(pc_cols) > 0, f"No PC name column found. Columns: {list(machines_df.columns)}"


def test_machines_no_duplicate_pcs(machines_df):
    pc_col = next((c for c in machines_df.columns if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    if pc_col:
        dupes = machines_df[pc_col].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate PC names in machines.csv"


def test_machines_no_all_null_rows(machines_df):
    all_null = machines_df.isnull().all(axis=1).sum()
    assert all_null == 0, f"Found {all_null} fully null rows in machines.csv"


# ---------------------------------------------------------------------------
# software.csv tests
# ---------------------------------------------------------------------------

def test_software_not_empty(software_df):
    assert len(software_df) > 0, "software.csv is empty"


def test_software_minimum_rows(software_df):
    """Expect at least 5,000 software install records for 520 PCs."""
    assert len(software_df) >= 5000, f"software.csv has only {len(software_df)} rows — expected at least 5,000"


def test_software_has_name_column(software_df):
    name_cols = [c for c in software_df.columns if c.lower() in ("software_name", "software", "app_name", "name", "product")]
    assert len(name_cols) > 0, f"No software name column found. Columns: {list(software_df.columns)}"


def test_software_has_pc_column(software_df):
    pc_cols = [c for c in software_df.columns if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")]
    assert len(pc_cols) > 0, f"No PC name column found in software.csv. Columns: {list(software_df.columns)}"


# ---------------------------------------------------------------------------
# hotfixes.csv tests
# ---------------------------------------------------------------------------

def test_hotfixes_not_empty(hotfixes_df):
    assert len(hotfixes_df) > 0, "hotfixes.csv is empty"


def test_hotfixes_minimum_rows(hotfixes_df):
    assert len(hotfixes_df) >= 1000, f"hotfixes.csv has only {len(hotfixes_df)} rows — expected at least 1,000"


# ---------------------------------------------------------------------------
# vulnerabilities.csv tests
# ---------------------------------------------------------------------------

def test_vulnerabilities_not_empty(vulnerabilities_df):
    assert len(vulnerabilities_df) > 0, "vulnerabilities.csv is empty"


def test_vulnerabilities_has_cve_column(vulnerabilities_df):
    cve_cols = [c for c in vulnerabilities_df.columns if c.lower() in ("cve_id", "cve", "vulnerability_id")]
    assert len(cve_cols) > 0, f"No CVE column found. Columns: {list(vulnerabilities_df.columns)}"
