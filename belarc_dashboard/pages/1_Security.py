"""Security page — patch compliance and per-PC / per-department risk scoring."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

import streamlit as st
from utils import db
from utils.ui import configure_page, empty_note, kpi_row, page_header, raw_data_expander

configure_page("Security")
page_header(
    "Security & Patch Compliance",
    "Where the risk is concentrated across the fleet — by department, by machine, and by CVE.",
)

# ---------------------------------------------------------------------------
# Patch compliance by department (v_patch_compliance)
# ---------------------------------------------------------------------------
st.subheader("Patch compliance by department")
if db.object_exists("v_patch_compliance"):
    pc_df = db.load_view("v_patch_compliance")
    if not empty_note(pc_df, "v_patch_compliance"):
        dept = db.first_col(pc_df, ["department", "dept", "business_unit"]) or pc_df.columns[0]
        # Prefer the patch-status breakdown counts (a stacked bar reads cleanly);
        # fall back to all numeric columns if those exact names aren't present.
        status_cols = [
            c
            for c in ["fully_patched", "partially_patched", "minimally_patched"]
            if c in pc_df.columns
        ]
        value_cols = status_cols or [
            c for c in db.numeric_cols(pc_df) if c not in (dept, "total_pcs", "full_patch_pct")
        ]
        if value_cols:
            long = pc_df.melt(
                id_vars=[dept], value_vars=value_cols, var_name="patch level", value_name="PCs"
            )
            long["patch level"] = long["patch level"].str.replace("_", " ").str.title()
            fig = px.bar(long, x=dept, y="PCs", color="patch level", barmode="stack")
            fig.update_layout(xaxis_title="", yaxis_title="PCs", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

        # Headline compliance % per department, sorted.
        pct_col = db.first_col(pc_df, ["full_patch_pct", "patch_compliance_pct"])
        if pct_col:
            fig2 = px.bar(
                pc_df.sort_values(pct_col), x=pct_col, y=dept, orientation="h",
                color=pct_col, color_continuous_scale="Greens",
            )
            fig2.update_layout(
                coloraxis_showscale=False, yaxis_title="", xaxis_title="% fully patched"
            )
            st.plotly_chart(fig2, use_container_width=True)
        raw_data_expander(pc_df)
else:
    st.info("View `v_patch_compliance` not found.")

st.divider()

# ---------------------------------------------------------------------------
# Security risk per PC (v_security_risk) + department roll-up (v_dept_risk)
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Highest-risk machines")
    if db.object_exists("v_security_risk"):
        risk = db.load_view("v_security_risk")
        if not empty_note(risk, "v_security_risk"):
            score = db.first_col(risk, ["risk_score", "security_score", "score", "risk"])
            name = db.first_col(risk, ["pc_name", "machine_name", "hostname", "computer_name"]) or risk.columns[0]
            if score:
                top = risk.sort_values(score, ascending=False).head(15)
                fig = px.bar(top, x=score, y=name, orientation="h",
                             color=score, color_continuous_scale="Reds")
                fig.update_layout(yaxis={"categoryorder": "total ascending"},
                                  coloraxis_showscale=False, yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

                kpi_row([
                    ("Avg risk score", round(float(risk[score].mean()), 1), None),
                    ("Max risk score", round(float(risk[score].max()), 1), None),
                    ("PCs above avg", int((risk[score] > risk[score].mean()).sum()), None),
                ])
            raw_data_expander(risk)
    else:
        st.info("View `v_security_risk` not found.")

with col_right:
    st.subheader("Vulnerabilities by department")
    if db.object_exists("v_dept_risk"):
        dr = db.load_view("v_dept_risk")
        if not empty_note(dr, "v_dept_risk"):
            dept = db.first_col(dr, ["department", "dept", "business_unit"]) or dr.columns[0]
            vcol = db.first_col(
                dr,
                ["total_vulns", "critical_vulns", "vuln_count", "vulnerability_count", "num_vulns"],
            )
            vcol = vcol or next(
                (c for c in db.numeric_cols(dr) if c != "total_pcs"),
                db.numeric_cols(dr)[0] if db.numeric_cols(dr) else None,
            )
            if vcol:
                fig = px.bar(dr.sort_values(vcol), x=vcol, y=dept, orientation="h",
                             color=vcol, color_continuous_scale="OrRd")
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Vulnerabilities")
                st.plotly_chart(fig, use_container_width=True)
            raw_data_expander(dr)
    else:
        st.info("View `v_dept_risk` not found.")

st.divider()

# ---------------------------------------------------------------------------
# Top CVEs across the fleet (from the vulnerabilities table)
# ---------------------------------------------------------------------------
st.subheader("Most common CVEs across the fleet")
if db.object_exists("vulnerabilities"):
    cols = db.columns_of("vulnerabilities")
    cve_col = next((c for c in cols if "cve" in c.lower()), None)
    sev_col = next((c for c in cols if "sever" in c.lower()), None)
    if cve_col:
        sql = f"SELECT {cve_col} AS cve, COUNT(*) AS affected_pcs FROM vulnerabilities GROUP BY {cve_col} ORDER BY affected_pcs DESC LIMIT 20"
        top_cves = db.run_query(sql)
        if not top_cves.empty:
            fig = px.bar(top_cves.sort_values("affected_pcs"), x="affected_pcs", y="cve",
                         orientation="h", color="affected_pcs", color_continuous_scale="Reds")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Affected PCs")
            st.plotly_chart(fig, use_container_width=True)
        if sev_col:
            sev = db.run_query(f"SELECT {sev_col} AS severity, COUNT(*) AS n FROM vulnerabilities GROUP BY {sev_col}")
            st.bar_chart(sev.set_index("severity"))
    else:
        st.info("No CVE column detected in `vulnerabilities`.")
else:
    st.info("Table `vulnerabilities` not found.")
