"""5_NLP.py — Software Categorization
Enterprise IT Asset Intelligence Platform

Visualises the NLP-assigned software categories produced by categorise_software.py.

Sections:
  1. Fleet-wide KPIs (total titles, coverage rate, top category, unknown count)
  2. Category distribution — donut + bar
  3. Category breakdown by department — stacked bar
  4. Top apps per category — tabbed view
  5. Unmanaged / unknown software table — filterable, flagged for review
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import db
from utils.ui import (
    configure_page,
    empty_note,
    kpi_row,
    page_header,
    raw_data_expander,
)

configure_page("NLP · Software Categories")
page_header(
    "Software Categorization",
    "How the installed software estate is classified — by type, by department, and by risk of unmanaged apps.",
)

# ---------------------------------------------------------------------------
# Guard: category column must exist
# ---------------------------------------------------------------------------
sw_cols = db.columns_of("software")
if "category" not in sw_cols:
    st.warning(
        "The `category` column is missing from the `software` table. "
        "Run **categorise_software.py** first, then refresh this page.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
# Detect column names defensively
_name_col = next(
    (c for c in sw_cols if c.lower() in ("software_name", "software", "app_name", "name", "product")),
    None,
)
_pc_col = next(
    (c for c in sw_cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")),
    None,
)

if _name_col is None or _pc_col is None:
    st.error("Could not detect required columns in `software` table. Check your schema.")
    st.stop()

@st.cache_data(ttl=600, show_spinner=False)
def load_software_with_categories() -> pd.DataFrame:
    return db.run_query(f"SELECT {_pc_col}, {_name_col}, category FROM software")


@st.cache_data(ttl=600, show_spinner=False)
def load_machines_dept() -> pd.DataFrame:
    cols = db.columns_of("machines")
    pc_col = next((c for c in cols if c.lower() in ("pc_name", "machine_name", "hostname", "computer_name")), None)
    dept_col = next((c for c in cols if c.lower() in ("department", "dept", "business_unit")), None)
    if pc_col and dept_col:
        return db.run_query(f"SELECT {pc_col} AS pc_name, {dept_col} AS department FROM machines")
    return pd.DataFrame()


sw = load_software_with_categories()
sw.columns = ["pc_name", "software_name", "category"]
sw["category"] = sw["category"].fillna("Unknown")

machines = load_machines_dept()

CATEGORY_ORDER = [
    "Security", "Productivity", "Development",
    "Finance/Analytics", "Remote Access", "Communication",
    "System/Utilities", "Unknown",
]

CATEGORY_COLOURS = {
    "Security":          "#e76f51",
    "Productivity":      "#2a9d8f",
    "Development":       "#1f3a5f",
    "Finance/Analytics": "#577590",
    "Remote Access":     "#f4a261",
    "Communication":     "#264653",
    "System/Utilities":  "#90a4ae",
    "Unknown":           "#b0bec5",
}

# Unique software titles (for coverage metrics)
unique_sw = sw.drop_duplicates(subset="software_name")
total_titles = len(unique_sw)
unknown_titles = int((unique_sw["category"] == "Unknown").sum())
classified_titles = total_titles - unknown_titles
coverage_pct = round(classified_titles / total_titles * 100, 1) if total_titles else 0
top_category = (
    unique_sw[unique_sw["category"] != "Unknown"]["category"]
    .value_counts()
    .idxmax()
    if classified_titles > 0
    else "—"
)

# ---------------------------------------------------------------------------
# Section 1 — KPIs
# ---------------------------------------------------------------------------
kpi_row([
    ("Unique software titles",  f"{total_titles:,}",       None),
    ("Classified titles",       f"{classified_titles:,}",  None),
    ("Classification coverage", f"{coverage_pct} %",       None),
    ("Unknown / unmanaged",     f"{unknown_titles:,}",     "Titles the pipeline could not categorise"),
])

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Fleet-wide category distribution
# ---------------------------------------------------------------------------
st.subheader("Category distribution — unique software titles")

cat_counts = (
    unique_sw.groupby("category")
    .size()
    .reindex(CATEGORY_ORDER, fill_value=0)
    .reset_index()
)
cat_counts.columns = ["category", "count"]
cat_counts = cat_counts[cat_counts["count"] > 0]

col_donut, col_bar = st.columns([1, 1])

with col_donut:
    fig_donut = px.pie(
        cat_counts,
        names="category",
        values="count",
        hole=0.55,
        color="category",
        color_discrete_map=CATEGORY_COLOURS,
        category_orders={"category": CATEGORY_ORDER},
    )
    fig_donut.update_traces(textposition="outside", textinfo="percent+label")
    fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_donut, use_container_width=True)

with col_bar:
    fig_bar = px.bar(
        cat_counts.sort_values("count", ascending=True),
        x="count",
        y="category",
        orientation="h",
        color="category",
        color_discrete_map=CATEGORY_COLOURS,
        labels={"count": "Unique titles", "category": ""},
        text="count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, xaxis_title="Unique software titles")
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Category breakdown by department
# ---------------------------------------------------------------------------
st.subheader("Software category mix by department")

if not machines.empty:
    sw_dept = sw.merge(machines, on="pc_name", how="left")
    sw_dept["department"] = sw_dept["department"].fillna("Unknown Dept")

    dept_cat = (
        sw_dept.groupby(["department", "category"])
        .size()
        .reset_index(name="installs")
    )

    fig_stack = px.bar(
        dept_cat,
        x="department",
        y="installs",
        color="category",
        color_discrete_map=CATEGORY_COLOURS,
        category_orders={"category": CATEGORY_ORDER},
        labels={"installs": "Install records", "department": "Department", "category": "Category"},
        barmode="stack",
    )
    fig_stack.update_layout(
        xaxis_tickangle=-30,
        legend_title_text="Category",
        height=450,
    )
    st.plotly_chart(fig_stack, use_container_width=True)
else:
    st.info("Department data not available — `machines` table missing or has no department column.")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Top apps per category (tabbed)
# ---------------------------------------------------------------------------
st.subheader("Top software titles per category")

available_cats = [c for c in CATEGORY_ORDER if c != "Unknown" and c in unique_sw["category"].values]
tabs = st.tabs(available_cats)

for tab, cat in zip(tabs, available_cats):
    with tab:
        # Count installs (rows in software table, not unique titles)
        top = (
            sw[sw["category"] == cat]
            .groupby("software_name")
            .size()
            .reset_index(name="install_count")
            .sort_values("install_count", ascending=False)
            .head(15)
        )
        if top.empty:
            st.info(f"No software found in category: {cat}")
            continue

        fig_top = px.bar(
            top.sort_values("install_count"),
            x="install_count",
            y="software_name",
            orientation="h",
            color_discrete_sequence=[CATEGORY_COLOURS.get(cat, "#2a9d8f")],
            labels={"install_count": "Installs", "software_name": ""},
            text="install_count",
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(
            showlegend=False,
            height=max(300, len(top) * 28),
            xaxis_title="Installs across fleet",
        )
        st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Unknown / unmanaged software
# ---------------------------------------------------------------------------
st.subheader("Unknown / unmanaged software")
st.caption(
    "These titles were not matched by keyword rules or the zero-shot classifier. "
    "Review for shadow IT, unlicensed tools, or niche applications that need manual categorization."
)

unknown_sw = sw[sw["category"] == "Unknown"].copy()

if unknown_sw.empty:
    st.success("No unknown software — full classification coverage achieved.")
else:
    # Count how many PCs each unknown title appears on
    unknown_summary = (
        unknown_sw.groupby("software_name")
        .agg(install_count=("pc_name", "count"), pcs=("pc_name", lambda x: ", ".join(sorted(x.unique()[:5]))))
        .reset_index()
        .sort_values("install_count", ascending=False)
    )
    unknown_summary.columns = ["Software Name", "Install Count", "Sample PCs (up to 5)"]

    # Colour-code by install count: higher spread = higher review priority
    def _highlight_high(row):
        if row["Install Count"] >= 10:
            return ["background-color: #ffeaea"] * len(row)
        elif row["Install Count"] >= 5:
            return ["background-color: #fff8e1"] * len(row)
        return [""] * len(row)

    search_term = st.text_input("Filter by software name", placeholder="Type to search …")
    filtered = unknown_summary
    if search_term:
        filtered = unknown_summary[
            unknown_summary["Software Name"].str.contains(search_term, case=False, na=False)
        ]

    kpi_row([
        ("Unknown titles", f"{len(unknown_summary):,}", None),
        ("Total installs",  f"{int(unknown_summary['Install Count'].sum()):,}", None),
    ])

    st.dataframe(
        filtered.style.apply(_highlight_high, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("🔴 Red = 10+ installs  |  🟡 Yellow = 5–9 installs  |  White = fewer than 5")

    raw_data_expander(unknown_sw, "View all unknown software rows")
