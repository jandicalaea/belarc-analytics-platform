# Enterprise IT Asset Intelligence Platform

An end-to-end data engineering and analytics portfolio project that simulates an enterprise IT asset management system — from raw data generation through ETL, database design, machine learning risk scoring, NLP software categorisation, and a RAG-powered chatbot, all surfaced through an interactive multi-page dashboard.

---

## Project Story

This project is inspired by a real-world PowerShell automation pipeline built to collect and parse Belarc Advisor hardware reports across a domain-joined Windows fleet. The pipeline was rebuilt and extended here as a full data engineering portfolio, demonstrating the complete journey from raw HTML reports to actionable business intelligence.

**The problem:** IT teams managing hundreds of PCs struggle to get a clear, centralised picture of their asset estate — patch compliance gaps, software sprawl, hardware aging, and CVE exposure all live in separate reports with no unified view.

**The solution:** A Python-powered platform that ingests Belarc-style HTML reports, parses and stores them in a structured SQLite database, and delivers insights through an interactive Streamlit dashboard with ML risk scoring, NLP software categorisation, and a natural language chatbot.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Generation | Python, Faker, BeautifulSoup |
| ETL & Parsing | Python, BeautifulSoup4, Pandas |
| Database | SQLite (via `sqlite3`), 4 tables, 8 indexes, 6 analytical views |
| Dashboard | Streamlit, Plotly |
| Machine Learning | Scikit-learn, XGBoost, SHAP, Joblib |
| NLP | HuggingFace Transformers (`facebook/bart-large-mnli`), regex rules |
| RAG Chatbot | ChromaDB, sentence-transformers, Groq API (llama-3.1-8b-instant) |
| CI/CD | GitHub Actions |
| Language | Python 3.10+ |

---

## Architecture

```
generate_mock_belarc.py          # Module 1 — generates 520 fake Belarc HTML reports
        ↓
parse_belarc.py                  # Module 2 — ETL parser → 5 clean CSVs
        ↓
load_database.py                 # Module 3 — loads CSVs into SQLite with views & indexes
        ↓
categorise_software.py           # Module 6 — NLP software categorisation
        ↓
lseg_assets.db (SQLite)
        ↓
belarc_dashboard/                # Module 4 — Streamlit dashboard
  Home.py                        # Overview: KPIs, OS distribution, dept breakdown
  pages/
    1_Security.py                # CVE exposure, patch compliance, risk heatmaps
    2_Hardware.py                # CPU, RAM, storage fleet analysis
    3_Software.py                # Software prevalence, install counts
    4_Predictions.py             # Module 5 — ML risk scoring (Random Forest / XGBoost + SHAP)
    5_NLP.py                     # Module 6 — NLP software category breakdown
    6_Chat.py                    # Module 7 — RAG chatbot (ChromaDB + Groq)
```

---

## Modules

### Module 1 — Mock Data Generator
Generates 520 realistic fake Belarc Advisor HTML reports covering PC name, department, CPU/RAM/storage specs, Windows OS version, installed enterprise software, Microsoft hotfixes, and CVE vulnerabilities.

### Module 2 — ETL Parser
Parses all 520 HTML files using BeautifulSoup into 5 clean CSVs:
- `machines.csv` — 520 rows (hardware, OS, patch status, vuln count)
- `software.csv` — ~8,400 rows (installed apps per PC)
- `hotfixes.csv` — ~5,400 rows (patches per PC)
- `vulnerabilities.csv` — CVE findings per PC
- `software_summary.csv` — aggregated app install counts

### Module 3 — Database Loader
Loads all CSVs into a normalised SQLite database with 4 core tables, 8 indexes, and 6 pre-built analytical views: `v_patch_compliance`, `v_security_risk`, `v_software_prevalence`, `v_os_distribution`, `v_dept_hardware`, `v_dept_risk`.

### Module 4 — Streamlit Dashboard
Multi-page interactive dashboard with KPI cards, Plotly charts, and department-level drilldowns across security, hardware, and software dimensions.

### Module 5 — ML Risk Scoring
Trains Random Forest vs XGBoost with 5-fold cross-validation on 16 engineered features. Saves the best model as `model.joblib`. The Predictions page shows risk distribution, SHAP feature importance, risk by department, and a colour-coded PC risk table.

### Module 6 — NLP Software Categorisation
Two-stage classification pipeline: keyword regex rules (covers ~90% of titles instantly) followed by `facebook/bart-large-mnli` zero-shot classification for unknowns. Classifies software into 7 categories: Security, Productivity, Development, Finance/Analytics, Remote Access, Communication, System/Utilities.

### Module 7 — RAG Chatbot
Converts all 520 PC records into text embeddings stored in ChromaDB. On each query, retrieves the top-8 most relevant chunks plus a fleet-wide summary, then calls `llama-3.1-8b-instant` via the Groq API to answer natural language questions about the fleet.

### Module 8 — GitHub + CI/CD
GitHub Actions workflow runs the ETL pipeline and pytest unit tests on every push.

---

## Screenshots

> _Add screenshots here after running the dashboard locally._

| Page | Description |
|---|---|
| Home | Fleet KPIs, OS distribution donut, department breakdown |
| Security | CVE heatmap, patch compliance bar, risk distribution |
| Hardware | CPU/RAM/storage histograms, aging analysis |
| Software | Top apps by install count, software prevalence |
| Predictions | ML risk scores, SHAP importance, risk by department |
| NLP | Software category donut, dept category stacked bar, unknown software table |
| Chat | RAG chatbot answering natural language fleet questions |

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/belarc-analytics-platform.git
cd belarc-analytics-platform
```

### 2. Install dependencies

```bash
pip install -r belarc_dashboard/requirements.txt
pip install beautifulsoup4 faker chromadb sentence-transformers groq
```

### 3. Generate mock data

```bash
python generate_mock_belarc.py
```

### 4. Run the ETL pipeline

```bash
python parse_belarc.py
python load_database.py
python categorise_software.py
```

### 5. Train the ML model

```bash
python train_model.py
```

### 6. Set environment variables

```bash
# Windows
setx GROQ_API_KEY "your-groq-api-key"

# Mac/Linux
export GROQ_API_KEY="your-groq-api-key"
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 7. Launch the dashboard

```bash
cd belarc_dashboard
streamlit run Home.py
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (for chatbot) | Free API key from console.groq.com |
| `ASSET_DB_PATH` | Optional | Custom path to `lseg_assets.db` |

---

## Project Structure

```
belarc-analytics-platform/
├── generate_mock_belarc.py
├── parse_belarc.py
├── load_database.py
├── categorise_software.py
├── train_model.py
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── tests/
│   ├── test_parser.py
│   └── test_db.py
├── mock_belarc_reports/          # generated — not committed
├── parsed_output/                # generated — not committed
├── lseg_assets.db                # generated — not committed
└── belarc_dashboard/
    ├── Home.py
    ├── model.joblib              # generated — not committed
    ├── requirements.txt
    ├── .streamlit/
    │   └── config.toml
    ├── utils/
    │   ├── __init__.py
    │   ├── db.py
    │   └── ui.py
    └── pages/
        ├── 1_Security.py
        ├── 2_Hardware.py
        ├── 3_Software.py
        ├── 4_Predictions.py
        ├── 5_NLP.py
        └── 6_Chat.py
```

---

## Key Design Decisions

- **SQLite over PostgreSQL** — keeps the project fully portable with zero infrastructure setup, appropriate for a 520-PC simulation
- **Keyword rules before NLP** — resolves ~90% of software titles instantly; zero-shot classification only runs on true unknowns, keeping runtime practical
- **Fleet summary injection** — every RAG query includes pre-computed aggregate statistics (total PCs, dept breakdown, patch status) so the chatbot answers aggregate questions correctly even when retrieved chunks are limited
- **XGBoost vs Random Forest** — both are trained and cross-validated; the better model is selected automatically and saved

---
