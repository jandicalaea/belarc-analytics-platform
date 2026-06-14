"""
train_model.py
==============
Module 5 — ML Risk Scoring

Trains a machine learning model to predict security risk level
per PC using features from assets.db.

OUTPUT:
    model.joblib        — trained model + preprocessor pipeline
    model_report.txt    — accuracy, ROC-AUC, classification report

USAGE:
    python train_model.py
    python train_model.py --db ./assets.db --out ./model.joblib
"""

import argparse
import sqlite3
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DB  = "./assets.db"
DEFAULT_OUT = "./model.joblib"
RANDOM_SEED = 42

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def load_features(db_path: str) -> pd.DataFrame:
    """
    Load and engineer features from assets.db.
    Returns a DataFrame with one row per PC including the target label.
    """
    conn = sqlite3.connect(db_path)

    # --- Base machine features -----------------------------------------------
    machines = pd.read_sql_query("""
        SELECT
            pc_name,
            department,
            os_name,
            cpu_cores,
            ram_gb,
            storage_gb,
            hotfix_count,
            software_count,
            vuln_count,
            has_critical_vuln,
            patch_status
        FROM machines
    """, conn)

    # --- Software risk features ----------------------------------------------
    # Count of known high-risk software categories per PC
    software = pd.read_sql_query("""
        SELECT
            pc_name,
            COUNT(*) AS total_sw,
            SUM(CASE WHEN LOWER(name) LIKE '%vnc%'
                      OR LOWER(name) LIKE '%teamviewer%'
                      OR LOWER(name) LIKE '%anydesk%'
                      OR LOWER(name) LIKE '%torrent%'
                      THEN 1 ELSE 0 END) AS risky_sw_count
        FROM software
        GROUP BY pc_name
    """, conn)

    # --- Hotfix recency feature ----------------------------------------------
    hotfixes = pd.read_sql_query("""
        SELECT pc_name, COUNT(*) AS patch_count
        FROM hotfixes
        GROUP BY pc_name
    """, conn)

    # --- Vulnerability severity features -------------------------------------
    vulns = pd.read_sql_query("""
        SELECT
            pc_name,
            COUNT(*) AS total_vulns,
            SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count,
            SUM(CASE WHEN severity = 'HIGH'     THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN severity = 'MEDIUM'   THEN 1 ELSE 0 END) AS medium_count
        FROM vulnerabilities
        GROUP BY pc_name
    """, conn)

    conn.close()

    # --- Merge all features --------------------------------------------------
    df = machines.copy()
    df = df.merge(software, on="pc_name", how="left")
    df = df.merge(hotfixes, on="pc_name", how="left")
    df = df.merge(vulns,    on="pc_name", how="left")

    # Fill NaN for PCs with no software/hotfix/vuln records
    df["total_sw"]       = df["total_sw"].fillna(0).astype(int)
    df["risky_sw_count"] = df["risky_sw_count"].fillna(0).astype(int)
    df["patch_count"]    = df["patch_count"].fillna(0).astype(int)
    df["total_vulns"]    = df["total_vulns"].fillna(0).astype(int)
    df["critical_count"] = df["critical_count"].fillna(0).astype(int)
    df["high_count"]     = df["high_count"].fillna(0).astype(int)
    df["medium_count"]   = df["medium_count"].fillna(0).astype(int)

    # --- Derived features ----------------------------------------------------
    # OS generation (Windows 11 = more secure than Windows 10)
    df["is_win11"] = df["os_name"].str.contains("11", na=False).astype(int)

    # Patch status as ordinal number
    patch_map = {"Full": 2, "Partial": 1, "Minimal": 0}
    df["patch_status_num"] = df["patch_status"].map(patch_map).fillna(0).astype(int)

    # Vulnerability weighted score (critical counts more)
    df["vuln_weighted"] = (
        df["critical_count"] * 4 +
        df["high_count"]     * 2 +
        df["medium_count"]   * 1
    )

    # RAM adequacy flag (< 8 GB is a risk indicator)
    df["low_ram"] = (df["ram_gb"] < 8).astype(int)

    # Software density (more software = larger attack surface)
    df["sw_density"] = df["total_sw"].fillna(0)

    # --- Target label --------------------------------------------------------
    # Derive risk level from the same formula used in v_security_risk view
    def compute_risk(row):
        score = (
            row["vuln_count"]       * 10 +
            row["has_critical_vuln"]* 30 +
            (20 if row["patch_status"] == "Minimal" else
             10 if row["patch_status"] == "Partial"  else 0)
        )
        if score >= 60: return "CRITICAL"
        if score >= 30: return "HIGH"
        if score >= 10: return "MEDIUM"
        return "LOW"

    df["risk_level"] = df.apply(compute_risk, axis=1)

    return df


def get_feature_columns():
    """Return the list of feature columns used for training."""
    return [
        "cpu_cores",
        "ram_gb",
        "storage_gb",
        "hotfix_count",
        "software_count",
        "vuln_count",
        "has_critical_vuln",
        "patch_status_num",
        "is_win11",
        "vuln_weighted",
        "critical_count",
        "high_count",
        "medium_count",
        "risky_sw_count",
        "low_ram",
        "sw_density",
    ]


# =============================================================================
# TRAINING
# =============================================================================

def train(db_path: str, out_path: str):
    print()
    print("=" * 55)
    print("  Module 5 — ML Risk Scoring Model Training")
    print("=" * 55)
    print(f"  Database : {Path(db_path).resolve()}")
    print(f"  Output   : {Path(out_path).resolve()}")
    print()

    # --- Load features -------------------------------------------------------
    print("  [1/5] Loading and engineering features...")
    df = load_features(db_path)
    print(f"        {len(df)} machines loaded")

    FEATURES = get_feature_columns()
    X = df[FEATURES].fillna(0)
    y = df["risk_level"]

    # Encode target labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"        Features : {len(FEATURES)}")
    print(f"        Classes  : {list(le.classes_)}")
    print()
    print("        Class distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"          {cls:<10} {count:>4} PCs  ({count/len(y)*100:.1f}%)")
    print()

    # --- Cross-validation comparison -----------------------------------------
    print("  [2/5] Cross-validating Random Forest vs XGBoost...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    rf_scores  = cross_val_score(rf,  X, y_enc, cv=cv, scoring="accuracy")
    xgb_scores = cross_val_score(xgb, X, y_enc, cv=cv, scoring="accuracy")

    rf_mean  = rf_scores.mean()
    xgb_mean = xgb_scores.mean()

    print(f"        Random Forest  CV accuracy: {rf_mean:.4f} ± {rf_scores.std():.4f}")
    print(f"        XGBoost        CV accuracy: {xgb_mean:.4f} ± {xgb_scores.std():.4f}")

    best_model = rf if rf_mean >= xgb_mean else xgb
    best_name  = "Random Forest" if rf_mean >= xgb_mean else "XGBoost"
    print(f"        Winner: {best_name}")
    print()

    # --- Train final model on full dataset -----------------------------------
    print(f"  [3/5] Training final {best_name} on full dataset...")
    best_model.fit(X, y_enc)

    y_pred = best_model.predict(X)
    acc    = accuracy_score(y_enc, y_pred)
    print(f"        Training accuracy : {acc:.4f}")
    print()

    # Classification report
    report = classification_report(y_enc, y_pred, target_names=le.classes_)
    print("        Classification Report:")
    for line in report.splitlines():
        print(f"          {line}")
    print()

    # --- Feature importances -------------------------------------------------
    print("  [4/5] Feature importances:")
    importances = best_model.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"        {feat:<22} {imp:.4f} {bar}")
    print()

    # --- Save model bundle ---------------------------------------------------
    print("  [5/5] Saving model bundle...")
    bundle = {
        "model":        best_model,
        "label_encoder":le,
        "features":     FEATURES,
        "model_name":   best_name,
        "trained_on":   len(df),
        "cv_accuracy":  max(rf_mean, xgb_mean),
        "classes":      list(le.classes_),
    }
    joblib.dump(bundle, out_path)
    print(f"        Saved to: {Path(out_path).resolve()}")

    # Save text report
    report_path = Path(out_path).parent / "model_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Trained on: {len(df)} machines\n")
        f.write(f"CV Accuracy: {max(rf_mean, xgb_mean):.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Feature Importances:\n")
        for feat, imp in feat_imp:
            f.write(f"  {feat:<22} {imp:.4f}\n")
    print(f"        Report : {report_path.resolve()}")

    print()
    print("=" * 55)
    print(f"  DONE — {best_name} model saved successfully")
    print("=" * 55)
    print()
    print("  Next step: open the dashboard Predictions page")
    print("    streamlit run Home.py")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML risk scoring model")
    parser.add_argument("--db",  default=DEFAULT_DB,  help="Path to assets.db")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output path for model.joblib")
    args = parser.parse_args()
    train(args.db, args.out)
