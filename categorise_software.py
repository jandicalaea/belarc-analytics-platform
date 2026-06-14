"""
Module 6 — NLP Software Categorisation
Enterprise IT Asset Intelligence Platform

Pipeline:
  1. Load all unique software names from the SQLite database.
  2. Apply keyword rules (fast, deterministic, covers ~80 % of fleet software).
  3. For anything still labelled "Unknown", run facebook/bart-large-mnli
     zero-shot classification (GPU if available, CPU otherwise).
  4. Write a `category` column back into the `software` table.
  5. Print a summary and save a categorisation_report.csv to parsed_output\.

Categories
  Security          — AV, EDR, VPN, firewall, encryption, identity
  Productivity      — Office suites, PDF tools, note-taking, project management
  Development       — IDEs, compilers, runtimes, version control, containers
  Finance/Analytics — BI tools, accounting, ERP, data platforms
  Remote Access     — RDP, VNC, screen sharing, remote support agents
  Communication     — Email clients, messaging, video conferencing
  System/Utilities  — OS components, drivers, runtimes, update agents
  Unknown           — Anything the pipeline cannot confidently classify
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_CANDIDATES = [
    os.environ.get("ASSET_DB_PATH", ""),
    str(Path(__file__).resolve().parent / "lseg_assets.db"),
    str(Path(__file__).resolve().parent.parent / "lseg_assets.db"),
    r"C:\Users\oli\Downloads\lseg_assets.db",
]
OUTPUT_DIR = Path(r"C:\Users\oli\Downloads\parsed_output")

HF_MODEL = "facebook/bart-large-mnli"
ZSC_BATCH_SIZE = 32          # items fed to the classifier at once
ZSC_CONFIDENCE_THRESHOLD = 0.55  # below this → stays "Unknown"

CATEGORIES = [
    "Security",
    "Productivity",
    "Development",
    "Finance/Analytics",
    "Remote Access",
    "Communication",
    "System/Utilities",
]

# ---------------------------------------------------------------------------
# Keyword rules  (pattern → category)
# Order matters — first match wins.
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, str]] = [
    # Security
    (r"(antivirus|anti.virus|endpoint.protect|edr|carbon.black|crowdstrike|"
     r"sentinel.?one|defender|malware|bitdefender|kaspersky|symantec|mcafee|"
     r"trend.micro|sophos|eset|cylance|darktrace|firewall|vpn|cisco.any|"
     r"forticlient|globalprotect|pulse.secure|zscaler|okta|duo.security|"
     r"cyberark|beyond.trust|secret.server|vault|encryption|bitlocker|"
     r"veracrypt|pgp|qualys|rapid7|nessus|tenable|splunk|arcsight|qradar|"
     r"solarwinds.security|varonis|digital.guardian|safetica|data.loss)",
     "Security"),

    # Remote Access
    (r"(remote.desktop|rdp|vnc|teamviewer|anydesk|logmein|bomgar|"
     r"beyond.trust.remote|dameware|screenconnect|connectwise.control|"
     r"splashtop|citrix|vmware.horizon|view.client|remote.utilities|"
     r"zoho.assist|goto.?assist|rescue|pcanywhere|chrome.remote)",
     "Remote Access"),

    # Communication
    (r"(microsoft.teams|slack|zoom|webex|skype|lync|jabber|discord|"
     r"mattermost|rocket\.chat|google.meet|gotomeeting|bluejeans|ringcentral|"
     r"outlook|thunderbird|lotus.notes|hcl.notes|postfix|sendmail|"
     r"whatsapp|signal|telegram.desktop)",
     "Communication"),

    # Development
    (r"(visual.studio|vscode|vs.code|intellij|pycharm|eclipse|netbeans|"
     r"android.studio|xcode|sublime.text|atom.editor|notepad\+\+|"
     r"git(?:hub|lab|kraken|lens)?|sourcetree|tortoisegit|svn|subversion|"
     r"docker|kubernetes|kubectl|helm|terraform|ansible|puppet|chef|"
     r"node\.?js|npm|yarn|python|java.?(?:jdk|jre|se)|\.net.?(?:sdk|runtime)|"
     r"ruby|go.lang|rust.?lang|cmake|make|gcc|clang|llvm|postman|insomnia|"
     r"dbeaver|datagrip|sequel.pro|pgadmin|redis|mongodb|mysql.?workbench|"
     r"wsl|windows.subsystem|cygwin|vagrant|virtualbox|vmware.workstation)",
     "Development"),

    # Finance / Analytics
    (r"(tableau|power.?bi|qlik|looker|microstrategy|cognos|business.objects|"
     r"sap.bo|crystal.reports|excel.?add|bloomberg|reuters|refinitiv|"
     r"eikon|matlab|spss|sas.(?:enterprise|base|studio)|stata|r.?studio|"
     r"anaconda|jupyter|orange.?data|alteryx|informatica|talend|"
     r"sap(?!.?gui.for.windows)|oracle.e.business|oracle.financials|"
     r"sage.(?:50|100|200|300|x3)|quickbooks|xero|freshbooks|netsuite|"
     r"workday|peoplesoft|hyperion|essbase|anaplan|adaptive.insights)",
     "Finance/Analytics"),

    # Productivity
    (r"(microsoft.office|office.?365|microsoft.365|word|excel|powerpoint|"
     r"onenote|visio|project.(?:professional|standard)|ms.project|"
     r"libreoffice|openoffice|wps.office|google.docs|g.?suite|"
     r"adobe.acrobat|foxit|nitro.pdf|pdf.?creator|pdf.?element|"
     r"cutepdf|pdfforge|docusign|adobe.sign|hellosign|"
     r"evernote|notion|confluence|sharepoint|box|dropbox|onedrive|"
     r"google.drive|trello|asana|monday\.com|basecamp|jira(?!.?service)|"
     r"todoist|microsoft.to.do|things|omnifocus|obsidian|roam.research|"
     r"snagit|greenshot|lightshot|camtasia|loom)",
     "Productivity"),

    # System / Utilities
    (r"(windows.update|microsoft.update|windows.?defender.update|"
     r"windows.?installer|microsoft.?c\+\+|visual.?c\+\+.?redistrib|"
     r"\.net.?framework|directx|windows.?sdk|wdk|driver|firmware|bios|"
     r"intel.?(?:me|management|amt|graphics.driver|chipset|rst|optane)|"
     r"nvidia.?(?:driver|geforce|cuda|control.panel)|amd.?(?:driver|radeon|"
     r"software)|realtek|synaptics|dell.?(?:command|update|bios|supportassist)|"
     r"hp.?(?:support|bios|connection|wolf)|lenovo.?(?:vantage|system.update|"
     r"bios)|microsoft.?(?:edge|webview|onedrive.update|malicious.software|"
     r"powershell|azure.?cli|sysinternals)|sccm|configmgr|landesk|ivanti|"
     r"bigfix|chocolatey|winget|7.?zip|winrar|winzip|peazip|"
     r"cpu.?z|gpu.?z|hwinfo|speccy|process.monitor|wireshark|putty|"
     r"winscp|filezilla|ccleaner|revo.uninstaller|malwarebytes|"
     r"glary.utilities|autoruns)",
     "System/Utilities"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), cat) for pat, cat in _RULES
]


def keyword_classify(name: str) -> str:
    """Return a category for `name` using compiled regex rules, or 'Unknown'."""
    for pattern, category in _COMPILED:
        if pattern.search(name):
            return category
    return "Unknown"


# ---------------------------------------------------------------------------
# Zero-shot classifier (HuggingFace)
# ---------------------------------------------------------------------------

def load_zsc_pipeline():
    """Load the zero-shot classification pipeline, GPU → CPU fallback."""
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    device_label = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    print(f"  Loading {HF_MODEL} on {device_label} …")

    classifier = pipeline(
        "zero-shot-classification",
        model=HF_MODEL,
        device=device,
        multi_label=False,
    )
    return classifier


def zsc_classify_batch(
    classifier,
    names: list[str],
) -> list[str]:
    """
    Classify a list of software names in batches.
    Returns a list of category strings, one per input name.
    """
    results: list[str] = []
    total = len(names)

    for start in range(0, total, ZSC_BATCH_SIZE):
        batch = names[start : start + ZSC_BATCH_SIZE]
        # Provide a short natural-language hypothesis template so the model
        # understands the classification task.
        outputs = classifier(
            batch,
            candidate_labels=CATEGORIES,
            hypothesis_template="This software is used for {}.",
        )
        # pipeline returns a dict for single input, list of dicts for batch
        if isinstance(outputs, dict):
            outputs = [outputs]

        for out in outputs:
            top_score: float = out["scores"][0]
            top_label: str = out["labels"][0]
            results.append(top_label if top_score >= ZSC_CONFIDENCE_THRESHOLD else "Unknown")

        done = min(start + ZSC_BATCH_SIZE, total)
        print(f"    Zero-shot: {done}/{total} classified …", end="\r")

    print()  # newline after progress
    return results


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def find_db() -> str:
    for candidate in DB_CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate assets.db / lseg_assets.db. "
        "Set the ASSET_DB_PATH environment variable or run load_database.py first."
    )


def ensure_category_column(conn: sqlite3.Connection) -> None:
    """Add `category` column to software table if it doesn't already exist."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(software)")]
    if "category" not in cols:
        conn.execute("ALTER TABLE software ADD COLUMN category TEXT DEFAULT 'Unknown'")
        conn.commit()
        print("  Added `category` column to software table.")
    else:
        print("  `category` column already exists — will overwrite values.")


def load_unique_software(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return a DataFrame of unique software names with the columns they appear under."""
    # Detect the software name column
    cols = [row[1] for row in conn.execute("PRAGMA table_info(software)")]
    name_col = next(
        (c for c in cols if c.lower() in ("software_name", "software", "app_name", "name", "product")),
        None,
    )
    if name_col is None:
        raise ValueError(f"Cannot find a software name column in `software`. Columns: {cols}")

    df = pd.read_sql_query(
        f"SELECT DISTINCT {name_col} AS software_name FROM software ORDER BY {name_col}",
        conn,
    )
    return df, name_col


def write_categories(
    conn: sqlite3.Connection,
    mapping: dict[str, str],
    name_col: str,
) -> int:
    """
    Bulk-update software.category based on software_name → category mapping.
    Returns number of rows updated.
    """
    cursor = conn.cursor()
    updated = 0
    for sw_name, category in mapping.items():
        cursor.execute(
            f"UPDATE software SET category = ? WHERE {name_col} = ?",
            (category, sw_name),
        )
        updated += cursor.rowcount
    conn.commit()
    return updated


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(mapping: dict[str, str], keyword_count: int, zsc_count: int) -> None:
    from collections import Counter

    cat_counts = Counter(mapping.values())
    total = len(mapping)

    print("\n" + "=" * 56)
    print("  SOFTWARE CATEGORISATION SUMMARY")
    print("=" * 56)
    print(f"  Unique software titles classified : {total:>6,}")
    print(f"  Via keyword rules                 : {keyword_count:>6,}")
    print(f"  Via zero-shot classifier          : {zsc_count:>6,}")
    print("-" * 56)
    print(f"  {'Category':<30} {'Titles':>8}  {'%':>6}")
    print("-" * 56)
    for cat in CATEGORIES + ["Unknown"]:
        n = cat_counts.get(cat, 0)
        pct = n / total * 100 if total else 0
        print(f"  {cat:<30} {n:>8,}  {pct:>5.1f}%")
    print("=" * 56 + "\n")


def save_report(mapping: dict[str, str], name_col: str, conn: sqlite3.Connection) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "categorisation_report.csv"

    # Pull full software table with pc_name for context
    df = pd.read_sql_query("SELECT * FROM software", conn)
    df["category"] = df[name_col].map(mapping).fillna("Unknown")
    df.to_csv(report_path, index=False)
    print(f"  Categorisation report saved → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n========================================")
    print("  Module 6 — NLP Software Categorisation")
    print("========================================\n")

    # 1. Connect to database
    db_path = find_db()
    print(f"  Database : {db_path}")
    conn = sqlite3.connect(db_path)

    # 2. Ensure category column exists
    ensure_category_column(conn)

    # 3. Load unique software names
    unique_sw_df, name_col = load_unique_software(conn)
    all_names: list[str] = unique_sw_df["software_name"].dropna().tolist()
    print(f"  Unique software titles found : {len(all_names):,}\n")

    # 4. Keyword classification pass
    print("Step 1/2 — Keyword rules …")
    t0 = time.time()
    mapping: dict[str, str] = {}
    for name in all_names:
        mapping[name] = keyword_classify(name)

    unknown_names = [n for n, c in mapping.items() if c == "Unknown"]
    keyword_count = len(all_names) - len(unknown_names)
    print(f"  Keyword pass complete in {time.time() - t0:.1f}s")
    print(f"  Classified by rules : {keyword_count:,}  |  Remaining Unknown : {len(unknown_names):,}\n")

    # 5. Zero-shot classification for unknowns
    zsc_count = 0
    if unknown_names:
        print(f"Step 2/2 — Zero-shot classification for {len(unknown_names):,} unknowns …")
        t1 = time.time()
        try:
            classifier = load_zsc_pipeline()
            zsc_labels = zsc_classify_batch(classifier, unknown_names)
            for name, label in zip(unknown_names, zsc_labels):
                mapping[name] = label
                if label != "Unknown":
                    zsc_count += 1
            print(f"  Zero-shot pass complete in {time.time() - t1:.1f}s")
            print(f"  Additional titles resolved : {zsc_count:,}\n")
        except ImportError as e:
            print(f"  [WARN] Could not load HuggingFace pipeline: {e}")
            print("  Skipping zero-shot step — unknowns will remain 'Unknown'.\n")
    else:
        print("Step 2/2 — No unknowns remain; skipping zero-shot classifier.\n")

    # 6. Write categories back to database
    print("Writing categories to database …")
    rows_updated = write_categories(conn, mapping, name_col)
    print(f"  Rows updated : {rows_updated:,}\n")

    # 7. Summary + report
    print_summary(mapping, keyword_count, zsc_count)
    save_report(mapping, name_col, conn)

    conn.close()
    print("  Done.\n")


if __name__ == "__main__":
    main()
