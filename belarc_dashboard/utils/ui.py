"""Shared UI helpers: page config, header, KPI cards, and a consistent Plotly theme."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

# Corporate-ish palette: deep navy + teal accent + neutral grays.
PALETTE = ["#1f3a5f", "#2a9d8f", "#e76f51", "#577590", "#90a4ae", "#264653"]

PAGE_ICON = "📊"


def configure_page(title: str) -> None:
    """Call once at the top of every page."""
    st.set_page_config(
        page_title=f"{title} · IT Asset Intelligence",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _install_plotly_theme()
    _sidebar_footer()


def _install_plotly_theme() -> None:
    template = pio.templates["plotly_white"]
    template.layout.colorway = PALETTE
    template.layout.font.family = "Inter, Segoe UI, sans-serif"
    template.layout.margin = dict(l=40, r=20, t=60, b=40)
    pio.templates["corporate"] = template
    pio.templates.default = "corporate"
    px.defaults.template = "corporate"
    px.defaults.color_discrete_sequence = PALETTE


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
    st.divider()


def kpi_row(metrics: list[tuple[str, str | int | float, str | None]]) -> None:
    """Render a row of KPI cards. Each metric is (label, value, delta_or_help)."""
    cols = st.columns(len(metrics))
    for col, (label, value, extra) in zip(cols, metrics):
        col.metric(label, value, help=extra if extra else None)


def empty_note(df: pd.DataFrame, name: str) -> bool:
    """Return True (and show a note) if a view/table came back empty."""
    if df is None or df.empty:
        st.info(f"No data returned from **{name}**. Make sure `load_database.py` has run.")
        return True
    return False


def raw_data_expander(df: pd.DataFrame, label: str = "View underlying data") -> None:
    with st.expander(label):
        st.dataframe(df, use_container_width=True)


def _sidebar_footer() -> None:
    with st.sidebar:
        st.markdown("---")
        st.caption("Enterprise IT Asset Intelligence Platform")
        st.caption("Data source: assets.db (SQLite)")
