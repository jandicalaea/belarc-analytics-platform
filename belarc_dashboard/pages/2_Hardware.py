"""Hardware page — fleet hardware composition and per-department capacity."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

import streamlit as st
from utils import db
from utils.ui import configure_page, empty_note, kpi_row, page_header, raw_data_expander

configure_page("Hardware")
page_header(
    "Hardware Overview",
    "RAM, CPU and storage distribution across the fleet, and how capacity is spread by department.",
)

machines = db.load_view("machines") if db.object_exists("machines") else pd.DataFrame()
if empty_note(machines, "machines"):
    st.stop()

ram_col = db.first_col(machines, ["ram_gb", "ram", "memory_gb", "total_ram_gb"])
cpu_col = db.first_col(machines, ["cpu_cores", "cores", "cpu_count", "logical_processors"])
cpu_name_col = db.first_col(machines, ["cpu", "cpu_model", "processor", "cpu_name"])
storage_col = db.first_col(machines, ["storage_gb", "disk_gb", "total_storage_gb", "storage"])
dept_col = db.first_col(machines, ["department", "dept", "business_unit"])

# --- KPIs -------------------------------------------------------------------
metrics = []
if ram_col:
    metrics.append(("Avg RAM (GB)", round(float(machines[ram_col].mean()), 1), None))
if cpu_col:
    metrics.append(("Avg CPU cores", round(float(machines[cpu_col].mean()), 1), None))
if storage_col:
    metrics.append(("Avg storage (GB)", round(float(machines[storage_col].mean()), 0), None))
if metrics:
    kpi_row(metrics)
    st.divider()

# --- Distributions ----------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    if ram_col:
        st.subheader("RAM distribution")
        fig = px.histogram(machines, x=ram_col, nbins=20)
        fig.update_layout(xaxis_title="RAM (GB)", yaxis_title="PCs")
        st.plotly_chart(fig, use_container_width=True)
with c2:
    if storage_col:
        st.subheader("Storage distribution")
        fig = px.histogram(machines, x=storage_col, nbins=20)
        fig.update_layout(xaxis_title="Storage (GB)", yaxis_title="PCs")
        st.plotly_chart(fig, use_container_width=True)

# --- CPU models -------------------------------------------------------------
if cpu_name_col:
    st.subheader("Top CPU models")
    cpu_counts = machines[cpu_name_col].value_counts().head(12).reset_index()
    cpu_counts.columns = ["CPU", "PCs"]
    fig = px.bar(cpu_counts.sort_values("PCs"), x="PCs", y="CPU", orientation="h")
    fig.update_layout(yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Department hardware (v_dept_hardware) ----------------------------------
st.subheader("Average hardware by department")
if db.object_exists("v_dept_hardware"):
    dh = db.load_view("v_dept_hardware")
    if not empty_note(dh, "v_dept_hardware"):
        dept = db.first_col(dh, ["department", "dept", "business_unit"]) or dh.columns[0]
        nums = db.numeric_cols(dh)
        if nums:
            metric = st.selectbox("Metric", nums, index=0)
            fig = px.bar(dh.sort_values(metric), x=metric, y=dept, orientation="h",
                         color=metric, color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        raw_data_expander(dh)
elif dept_col and ram_col:
    # Fallback: compute the roll-up ourselves if the view is missing.
    agg = machines.groupby(dept_col)[[c for c in [ram_col, cpu_col, storage_col] if c]].mean().reset_index()
    st.dataframe(agg, use_container_width=True)
else:
    st.info("View `v_dept_hardware` not found.")
