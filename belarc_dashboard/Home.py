"""
Enterprise IT Asset Intelligence Platform — Streamlit dashboard entry point.

Run with:
    streamlit run Home.py

This is the Overview landing page. The Security / Hardware / Software /
Predictions pages live in the pages/ directory and are picked up automatically
by Streamlit's multi-page router.
"""

from __future__ import annotations

import pandas as pd

import streamlit as st
from utils import db
from utils.ui import (
    configure_page,
    empty_note,
    kpi_row,
    page_header,
    raw_data_expander,
)
import plotly.express as px

configure_page("Overview")
page_header(
    "Enterprise IT Asset Intelligence Platform",
    "Fleet-wide view of 500+ enterprise PCs — hardware, patching, software and security posture.",
)

# --- Connectivity / schema sanity check ------------------------------------
objects = db.list_objects()
if objects.empty:
    st.error("The database contains no tables or views. Run `load_database.py` first.")
    st.stop()

# --- Headline KPIs ----------------------------------------------------------
machines = db.load_view("machines") if db.object_exists("machines") else pd.DataFrame()

total_pcs = len(machines)

# vulnerability count column is detected defensively
vuln_col = db.first_col(machines, ["vuln_count", "vulnerability_count", "num_vulns", "vulnerabilities"])
total_vulns = int(machines[vuln_col].sum()) if vuln_col else None

# patch status column
patch_col = db.first_col(
    machines, ["patch_status", "compliance_status", "is_compliant", "compliant"]
)

compliant_pct = None
if patch_col and not machines.empty:
    series = machines[patch_col].astype(str).str.lower()
    # "Full" is the fully-patched state in this schema; also accept generic synonyms.
    compliant_mask = series.isin(
        ["full", "compliant", "up to date", "up-to-date", "1", "true", "yes"]
    )
    compliant_pct = round(100 * compliant_mask.mean(), 1)

software_count = (
    db.run_query("SELECT COUNT(*) AS n FROM software").iloc[0]["n"]
    if db.object_exists("software")
    else None
)

metrics = [("Total PCs", f"{total_pcs:,}", "One row per machine")]
if compliant_pct is not None:
    metrics.append(("Patch compliance", f"{compliant_pct}%", "Share of fleet up to date"))
if total_vulns is not None:
    metrics.append(("Open vulnerabilities", f"{total_vulns:,}", "Sum across the fleet"))
if software_count is not None:
    metrics.append(("Installed software rows", f"{int(software_count):,}", "Across all PCs"))

kpi_row(metrics)
st.divider()

# --- OS distribution + departments -----------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Operating system distribution")
    if db.object_exists("v_os_distribution"):
        os_df = db.load_view("v_os_distribution")
        if not empty_note(os_df, "v_os_distribution"):
            label = db.first_col(os_df, ["os", "os_name", "operating_system", "os_version"]) or os_df.columns[0]
            value = db.first_col(os_df, ["count", "pc_count", "n", "num_pcs"]) or db.numeric_cols(os_df)[0]
            fig = px.pie(os_df, names=label, values=value, hole=0.5)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
            raw_data_expander(os_df)
    else:
        st.info("View `v_os_distribution` not found.")

with right:
    st.subheader("PCs by department")
    if not machines.empty:
        dept_col = db.first_col(machines, ["department", "dept", "business_unit"])
        if dept_col:
            counts = machines[dept_col].value_counts().reset_index()
            counts.columns = ["Department", "PCs"]
            fig = px.bar(counts, x="PCs", y="Department", orientation="h")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No department column found in `machines`.")
    else:
        st.info("Table `machines` not found.")

st.divider()

# --- Schema overview --------------------------------------------------------
st.subheader("Database objects")
st.caption("Tables and pre-built analytical views available to this dashboard.")
tables = objects[objects["type"] == "table"]["name"].tolist()
views = objects[objects["type"] == "view"]["name"].tolist()
c1, c2 = st.columns(2)
c1.write("**Tables**")
c1.write(tables or "None")
c2.write("**Views**")
c2.write(views or "None")
