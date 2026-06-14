"""Software page — fleet-wide software prevalence and per-PC inventory lookup."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

import streamlit as st
from utils import db
from utils.ui import configure_page, empty_note, kpi_row, page_header, raw_data_expander

configure_page("Software")
page_header(
    "Software Inventory",
    "What is installed across the estate, how widely, and what a single machine is running.",
)

# ---------------------------------------------------------------------------
# Prevalence (v_software_prevalence, falls back to software_summary)
# ---------------------------------------------------------------------------
st.subheader("Most prevalent software across the fleet")

prevalence_source = None
for candidate in ("v_software_prevalence", "software_summary"):
    if db.object_exists(candidate):
        prevalence_source = candidate
        break

if prevalence_source:
    prev = db.load_view(prevalence_source)
    if not empty_note(prev, prevalence_source):
        name = db.first_col(prev, ["software", "software_name", "app_name", "name", "product"]) or prev.columns[0]
        count = db.first_col(prev, ["install_count", "count", "installs", "num_pcs", "pc_count", "n"])
        count = count or (db.numeric_cols(prev)[0] if db.numeric_cols(prev) else None)
        if count:
            top_n = st.slider("Show top N", min_value=5, max_value=40, value=20, step=5)
            top = prev.sort_values(count, ascending=False).head(top_n)
            fig = px.bar(top.sort_values(count), x=count, y=name, orientation="h",
                         color=count, color_continuous_scale="Teal")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Installs")
            fig.update_layout(height=max(400, top_n * 22))
            st.plotly_chart(fig, use_container_width=True)
        raw_data_expander(prev)
else:
    st.info("Neither `v_software_prevalence` nor `software_summary` was found.")

st.divider()

# ---------------------------------------------------------------------------
# Per-PC software lookup (software table)
# ---------------------------------------------------------------------------
st.subheader("Per-machine software lookup")
if db.object_exists("software"):
    cols = db.columns_of("software")
    pc_col = next((c for c in cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    sw_col = next((c for c in cols if c.lower() in ("software", "software_name", "app_name", "name", "product")), None)

    if pc_col:
        pcs = db.run_query(f"SELECT DISTINCT {pc_col} AS pc FROM software ORDER BY {pc_col}")["pc"].tolist()
        selected = st.selectbox("Select a machine", pcs)
        if selected:
            installed = db.run_query(
                f"SELECT * FROM software WHERE {pc_col} = ? ORDER BY {sw_col or pc_col}",
                params=(selected,),
            )
            kpi_row([("Apps installed", len(installed), None)])
            st.dataframe(installed, use_container_width=True)
    else:
        st.info("No PC-name column detected in `software`.")
else:
    st.info("Table `software` not found.")

st.divider()

# ---------------------------------------------------------------------------
# Hotfix / patch coverage (hotfixes table)
# ---------------------------------------------------------------------------
st.subheader("Patch / hotfix coverage")
if db.object_exists("hotfixes"):
    total_hotfixes = db.run_query("SELECT COUNT(*) AS n FROM hotfixes").iloc[0]["n"]
    cols = db.columns_of("hotfixes")
    pc_col = next((c for c in cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    kpis = [("Total hotfix records", f"{int(total_hotfixes):,}", None)]
    if pc_col:
        per_pc = db.run_query(f"SELECT {pc_col} AS pc, COUNT(*) AS n FROM hotfixes GROUP BY {pc_col}")
        kpis.append(("Avg hotfixes / PC", round(float(per_pc["n"].mean()), 1), None))
        kpi_row(kpis)
        fig = px.histogram(per_pc, x="n", nbins=20)
        fig.update_layout(xaxis_title="Hotfixes per PC", yaxis_title="PCs")
        st.plotly_chart(fig, use_container_width=True)
    else:
        kpi_row(kpis)
else:
    st.info("Table `hotfixes` not found.")
