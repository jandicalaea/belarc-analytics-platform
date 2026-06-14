"""
pages/4_Predictions.py
======================
Module 5 — ML Risk Scoring Dashboard Page

Loads the trained model (model.joblib) and scores all machines live.
Shows:
  - Risk distribution chart
  - SHAP feature importance (global)
  - Per-machine risk scores table with highlighting
  - Department-level risk breakdown
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure utils is importable when running from pages/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import db
from utils.ui import configure_page, empty_note, page_header

configure_page("Predictions")
page_header(
    "Predictive Risk Scoring",
    "Machine-learning risk scores per PC with SHAP explainability.",
)

# =============================================================================
# LOCATE MODEL
# =============================================================================

# Search for model.joblib relative to the dashboard root
_HERE    = Path(__file__).resolve().parents[1]
_CANDIDATES = [
    _HERE / "model.joblib",
    _HERE.parent / "model.joblib",
    Path("model.joblib"),
]

model_path = next((p for p in _CANDIDATES if p.exists()), None)

if model_path is None:
    st.error(
        "**model.joblib not found.**\n\n"
        "Run the training script first:\n"
        "```\npython train_model.py --db ./assets.db\n```\n"
        "Then place `model.joblib` next to `Home.py`."
    )
    st.stop()

# =============================================================================
# LOAD MODEL BUNDLE
# =============================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)

bundle = load_model(str(model_path))
model  = bundle["model"]
le     = bundle["label_encoder"]
FEATURES = bundle["features"]
MODEL_NAME  = bundle["model_name"]
CV_ACCURACY = bundle["cv_accuracy"]

# =============================================================================
# LOAD + SCORE MACHINES
# =============================================================================

@st.cache_data(ttl=300, show_spinner="Scoring machines...")
def score_machines() -> pd.DataFrame:
    """Load feature data from DB and run live model predictions."""

    machines = db.run_query("""
        SELECT pc_name, department, os_name, cpu_cores, ram_gb,
               storage_gb, hotfix_count, software_count,
               vuln_count, has_critical_vuln, patch_status
        FROM machines
    """)

    software = db.run_query("""
        SELECT pc_name,
               COUNT(*) AS total_sw,
               SUM(CASE WHEN LOWER(name) LIKE '%vnc%'
                         OR LOWER(name) LIKE '%teamviewer%'
                         OR LOWER(name) LIKE '%anydesk%'
                         OR LOWER(name) LIKE '%torrent%'
                         THEN 1 ELSE 0 END) AS risky_sw_count
        FROM software GROUP BY pc_name
    """)

    vulns = db.run_query("""
        SELECT pc_name,
               COUNT(*) AS total_vulns,
               SUM(CASE WHEN severity='CRITICAL' THEN 1 ELSE 0 END) AS critical_count,
               SUM(CASE WHEN severity='HIGH'     THEN 1 ELSE 0 END) AS high_count,
               SUM(CASE WHEN severity='MEDIUM'   THEN 1 ELSE 0 END) AS medium_count
        FROM vulnerabilities GROUP BY pc_name
    """)

    df = machines.merge(software, on="pc_name", how="left")
    df = df.merge(vulns,    on="pc_name", how="left")

    # Fill missing values
    for col in ["total_sw","risky_sw_count","critical_count",
                "high_count","medium_count","total_vulns"]:
        df[col] = df[col].fillna(0).astype(int)

    # Engineer same features as training
    df["is_win11"]        = df["os_name"].str.contains("11", na=False).astype(int)
    patch_map             = {"Full": 2, "Partial": 1, "Minimal": 0}
    df["patch_status_num"]= df["patch_status"].map(patch_map).fillna(0).astype(int)
    df["vuln_weighted"]   = df["critical_count"]*4 + df["high_count"]*2 + df["medium_count"]
    df["low_ram"]         = (df["ram_gb"] < 8).astype(int)
    df["sw_density"]      = df["total_sw"].fillna(0)

    # Score
    X           = df[FEATURES].fillna(0)
    y_pred_enc  = model.predict(X)
    y_pred_prob = model.predict_proba(X)
    y_pred_lbl  = le.inverse_transform(y_pred_enc)

    df["risk_level"]    = y_pred_lbl
    df["risk_score_pct"]= (y_pred_prob.max(axis=1) * 100).round(1)

    # Max probability for the highest-risk class
    classes = list(le.classes_)
    for cls in classes:
        idx = list(le.classes_).index(cls)
        df[f"prob_{cls.lower()}"] = (y_pred_prob[:, idx] * 100).round(1)

    return df

scored = score_machines()

# =============================================================================
# RISK COLOUR MAP
# =============================================================================

RISK_COLORS = {
    "CRITICAL": "#cc0000",
    "HIGH":     "#ff6600",
    "MEDIUM":   "#ffaa00",
    "LOW":      "#2da44e",
}

# =============================================================================
# SECTION 1 — MODEL INFO BAR
# =============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", MODEL_NAME)
c2.metric("CV Accuracy", f"{CV_ACCURACY:.1%}")
c3.metric("Machines Scored", f"{len(scored):,}")
c4.metric("High / Critical", str(scored["risk_level"].isin(["HIGH","CRITICAL"]).sum()))

st.divider()

# =============================================================================
# SECTION 2 — RISK DISTRIBUTION
# =============================================================================

st.subheader("Risk level distribution")

risk_counts = (
    scored["risk_level"]
    .value_counts()
    .reindex(["LOW","MEDIUM","HIGH","CRITICAL"])
    .fillna(0)
    .reset_index()
)
risk_counts.columns = ["Risk Level", "Count"]
risk_counts["Color"] = risk_counts["Risk Level"].map(RISK_COLORS)

col_left, col_right = st.columns(2)

with col_left:
    fig_bar = px.bar(
        risk_counts,
        x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_map=RISK_COLORS,
        text="Count",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of PCs")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    fig_pie = px.pie(
        risk_counts,
        names="Risk Level", values="Count",
        color="Risk Level",
        color_discrete_map=RISK_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 3 — SHAP FEATURE IMPORTANCE
# =============================================================================

st.subheader("Feature importance (SHAP)")
st.caption("Which features drive the model's risk predictions the most.")

@st.cache_data(show_spinner="Computing SHAP values...")
def get_shap_values():
    try:
        import shap
        X_sample = scored[FEATURES].fillna(0)
        # Use TreeExplainer - works with both RF and XGBoost
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        # For multi-class, shap_values is a list — take mean absolute across classes
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values)
        mean_importance = mean_abs.mean(axis=0)
        return dict(zip(FEATURES, mean_importance))
    except Exception:
        # Fallback to built-in feature importances
        return dict(zip(FEATURES, model.feature_importances_))

shap_dict = get_shap_values()

shap_df = (
    pd.DataFrame.from_dict(shap_dict, orient="index", columns=["importance"])
    .sort_values("importance", ascending=True)
    .reset_index()
    .rename(columns={"index": "feature"})
)

# Friendly feature name mapping
name_map = {
    "hotfix_count":      "Hotfix count",
    "patch_status_num":  "Patch status",
    "vuln_count":        "Vulnerability count",
    "has_critical_vuln": "Has critical vuln",
    "vuln_weighted":     "Vuln weighted score",
    "critical_count":    "Critical CVEs",
    "high_count":        "High CVEs",
    "medium_count":      "Medium CVEs",
    "cpu_cores":         "CPU cores",
    "ram_gb":            "RAM (GB)",
    "storage_gb":        "Storage (GB)",
    "is_win11":          "Is Windows 11",
    "software_count":    "Software count",
    "sw_density":        "Software density",
    "risky_sw_count":    "Risky software",
    "low_ram":           "Low RAM flag",
}
shap_df["feature_label"] = shap_df["feature"].map(name_map).fillna(shap_df["feature"])

fig_shap = px.bar(
    shap_df,
    x="importance", y="feature_label",
    orientation="h",
    color="importance",
    color_continuous_scale="Blues",
    labels={"importance": "Mean |SHAP value|", "feature_label": ""},
)
fig_shap.update_layout(coloraxis_showscale=False, yaxis_title="")
st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 4 — RISK BY DEPARTMENT
# =============================================================================

st.subheader("Risk distribution by department")

dept_risk = (
    scored.groupby(["department","risk_level"])
    .size()
    .reset_index(name="count")
)

fig_dept = px.bar(
    dept_risk,
    x="count", y="department",
    color="risk_level",
    orientation="h",
    color_discrete_map=RISK_COLORS,
    barmode="stack",
    labels={"count": "PCs", "department": "", "risk_level": "Risk Level"},
)
fig_dept.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_dept, use_container_width=True)

st.divider()

# =============================================================================
# SECTION 5 — PC RISK TABLE
# =============================================================================

st.subheader("All machines — predicted risk scores")
st.caption("Filter by department or risk level. Click column headers to sort.")

col_dept, col_risk, col_search = st.columns([2, 2, 3])

with col_dept:
    dept_options = ["All"] + sorted(scored["department"].dropna().unique().tolist())
    selected_dept = st.selectbox("Department", dept_options)

with col_risk:
    risk_options = ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    selected_risk = st.selectbox("Risk level", risk_options)

with col_search:
    search = st.text_input("Search PC name", placeholder="e.g. jsmith-pc")

# Apply filters
filtered = scored.copy()
if selected_dept != "All":
    filtered = filtered[filtered["department"] == selected_dept]
if selected_risk != "All":
    filtered = filtered[filtered["risk_level"] == selected_risk]
if search:
    filtered = filtered[filtered["pc_name"].str.contains(search, case=False, na=False)]

# Display columns
display_cols = [
    "pc_name", "department", "os_name", "ram_gb",
    "hotfix_count", "vuln_count", "has_critical_vuln",
    "patch_status", "risk_level", "risk_score_pct",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.caption(f"Showing {len(filtered):,} of {len(scored):,} machines")

# Colour the risk_level column
def highlight_risk(val):
    colors = {
        "CRITICAL": "background-color:#cc0000;color:white;font-weight:bold",
        "HIGH":     "background-color:#ff6600;color:white;font-weight:bold",
        "MEDIUM":   "background-color:#ffaa00;color:black;font-weight:bold",
        "LOW":      "background-color:#2da44e;color:white",
    }
    return colors.get(val, "")

styled = (
    filtered[display_cols]
    .rename(columns={
        "pc_name":         "PC Name",
        "department":      "Department",
        "os_name":         "OS",
        "ram_gb":          "RAM (GB)",
        "hotfix_count":    "Patches",
        "vuln_count":      "Vulns",
        "has_critical_vuln":"Critical Vuln",
        "patch_status":    "Patch Status",
        "risk_level":      "Risk Level",
        "risk_score_pct":  "Confidence %",
    })
    .style.applymap(highlight_risk, subset=["Risk Level"])
)

st.dataframe(styled, use_container_width=True, height=420)
