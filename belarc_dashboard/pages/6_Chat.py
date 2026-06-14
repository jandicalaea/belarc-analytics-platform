"""6_Chat.py — RAG Chatbot
Enterprise IT Asset Intelligence Platform

Architecture:
  - SQLite rows converted to text documents and embedded using
    sentence-transformers (all-MiniLM-L6-v2, runs fully locally).
  - Embeddings stored in a persistent ChromaDB collection on disk.
  - On each user query: retrieve top-k relevant documents, build a
    context-stuffed prompt, call llama-3.1-8b-instant via the Groq API (free tier).
  - Full chat history maintained in st.session_state for multi-turn Q&A.

First run:  builds the vector store (~30s for 520 PCs).
Subsequent runs: loads from disk instantly.
"""

from __future__ import annotations

import os
import sqlite3
import textwrap
from pathlib import Path

import streamlit as st

from utils import db
from utils.ui import configure_page, page_header

configure_page("Chat · RAG Assistant")
page_header(
    "IT Asset Q&A",
    "Ask natural language questions about the fleet — powered by RAG and Gemini.",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "chroma_store")
COLLECTION_NAME = "it_assets"
EMBED_MODEL = "all-MiniLM-L6-v2"          # local, no API key needed
TOP_K = 8                                   # retrieved chunks per query
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

SYSTEM_PROMPT = textwrap.dedent("""
    You are an IT Asset Intelligence Assistant for an enterprise fleet of Windows PCs.
    You have access to retrieved records from a SQLite database containing information
    about PC hardware, installed software, patch compliance, and CVE vulnerabilities.

    Answer questions concisely and accurately based ONLY on the context provided.
    If the context does not contain enough information to answer, say so clearly.
    When listing PCs or software, use bullet points.
    Do not make up data that is not in the context.
""").strip()

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
_missing = []
try:
    import chromadb
except ImportError:
    _missing.append("chromadb")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _missing.append("sentence-transformers")
try:
    from groq import Groq
except ImportError:
    _missing.append("groq")

if _missing:
    st.error(
        f"Missing packages: `{', '.join(_missing)}`\n\n"
        f"Run: `pip install {' '.join(_missing)}`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.warning(
        "**GROQ_API_KEY** environment variable not set.\n\n"
        "Set it with `setx GROQ_API_KEY \"your-key-here\"` and restart the terminal.",
        icon="🔑",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Build text documents from the database
# ---------------------------------------------------------------------------

def _resolve_db() -> str:
    path = db.resolve_db_path()
    if not path:
        st.error("Database not found. Run `load_database.py` first.")
        st.stop()
    return path


@st.cache_data(ttl=600, show_spinner=False)
def load_fleet_summary() -> str:
    """Build a concise fleet-wide summary injected into every prompt."""
    db_path = _resolve_db()
    conn = sqlite3.connect(db_path)

    m_cols = [r[1] for r in conn.execute("PRAGMA table_info(machines)")]
    pc_col    = next((c for c in m_cols if c.lower() in ("pc_name","machine_name","hostname","computer_name")), None)
    dept_col  = next((c for c in m_cols if c.lower() in ("department","dept","business_unit")), None)
    os_col    = next((c for c in m_cols if c.lower() in ("os_name","os","operating_system","windows_version")), None)
    patch_col = next((c for c in m_cols if c.lower() in ("patch_status","patch_compliance","update_status")), None)
    vuln_col  = next((c for c in m_cols if c.lower() in ("vuln_count","vulnerability_count","cve_count")), None)

    total_pcs = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]

    dept_counts = ""
    if dept_col and pc_col:
        rows = conn.execute(
            f"SELECT {dept_col}, COUNT(*) as n FROM machines GROUP BY {dept_col} ORDER BY n DESC"
        ).fetchall()
        dept_counts = "\n".join(f"  - {r[0]}: {r[1]} PCs" for r in rows)

    os_counts = ""
    if os_col:
        rows = conn.execute(
            f"SELECT {os_col}, COUNT(*) as n FROM machines GROUP BY {os_col} ORDER BY n DESC LIMIT 10"
        ).fetchall()
        os_counts = "\n".join(f"  - {r[0]}: {r[1]} PCs" for r in rows)

    patch_counts = ""
    if patch_col:
        rows = conn.execute(
            f"SELECT {patch_col}, COUNT(*) as n FROM machines GROUP BY {patch_col} ORDER BY n DESC"
        ).fetchall()
        patch_counts = "\n".join(f"  - {r[0]}: {r[1]} PCs" for r in rows)

    total_vulns = 0
    if vuln_col:
        total_vulns = conn.execute(f"SELECT SUM({vuln_col}) FROM machines").fetchone()[0] or 0

    total_sw = conn.execute("SELECT COUNT(*) FROM software").fetchone()[0]
    unique_sw = conn.execute("SELECT COUNT(DISTINCT software_name) FROM software").fetchone()[0] if "software_name" in [r[1] for r in conn.execute("PRAGMA table_info(software)")] else "N/A"

    conn.close()

    summary = f"""=== FLEET-WIDE SUMMARY (use these facts to answer aggregate questions) ===
Total PCs in fleet: {total_pcs}
Total software install records: {total_sw}
Unique software titles: {unique_sw}
Total vulnerabilities (sum across all PCs): {total_vulns}

PCs by department:
{dept_counts if dept_counts else '  N/A'}

PCs by OS:
{os_counts if os_counts else '  N/A'}

PCs by patch status:
{patch_counts if patch_counts else '  N/A'}
=== END FLEET SUMMARY ==="""

    return summary


def _load_documents(db_path: str) -> list[str]:
    """Convert each machine row + its software/vulns into a text chunk."""
    conn = sqlite3.connect(db_path)
    docs: list[str] = []

    # Detect column names
    m_cols = [r[1] for r in conn.execute("PRAGMA table_info(machines)")]
    pc_col  = next((c for c in m_cols if c.lower() in ("pc_name","machine_name","hostname","computer_name")), None)
    dept_col = next((c for c in m_cols if c.lower() in ("department","dept","business_unit")), None)
    os_col   = next((c for c in m_cols if c.lower() in ("os_name","os","operating_system","windows_version")), None)
    ram_col  = next((c for c in m_cols if c.lower() in ("ram_gb","ram","memory_gb")), None)
    cpu_col  = next((c for c in m_cols if c.lower() in ("cpu","cpu_name","processor")), None)
    patch_col = next((c for c in m_cols if c.lower() in ("patch_status","patch_compliance","update_status")), None)
    vuln_col  = next((c for c in m_cols if c.lower() in ("vuln_count","vulnerability_count","cve_count")), None)

    sw_cols = [r[1] for r in conn.execute("PRAGMA table_info(software)")]
    sw_pc   = next((c for c in sw_cols if c.lower() in ("pc_name","machine_name","hostname","computer_name")), None)
    sw_name = next((c for c in sw_cols if c.lower() in ("software_name","software","app_name","name","product")), None)
    sw_cat  = "category" if "category" in sw_cols else None

    vul_cols = [r[1] for r in conn.execute("PRAGMA table_info(vulnerabilities)")]
    vul_pc   = next((c for c in vul_cols if c.lower() in ("pc_name","machine_name","hostname","computer_name")), None)
    vul_cve  = next((c for c in vul_cols if c.lower() in ("cve_id","cve","vulnerability_id")), None)
    vul_sev  = next((c for c in vul_cols if c.lower() in ("severity","cvss_severity","risk_level")), None)

    machines = conn.execute(f"SELECT * FROM machines").fetchall()
    m_col_names = [d[0] for d in conn.execute("PRAGMA table_info(machines)")]

    for row in machines:
        m = dict(zip(m_col_names, row))
        pc = m.get(pc_col, "Unknown PC")

        # Software list for this PC
        sw_parts: list[str] = []
        if sw_pc and sw_name:
            cat_select = f", {sw_cat}" if sw_cat else ""
            sw_rows = conn.execute(
                f"SELECT {sw_name}{cat_select} FROM software WHERE {sw_pc} = ?", (pc,)
            ).fetchall()
            for sr in sw_rows[:20]:  # cap to avoid huge chunks
                entry = sr[0]
                if sw_cat and len(sr) > 1 and sr[1]:
                    entry += f" [{sr[1]}]"
                sw_parts.append(entry)

        # CVEs for this PC
        cve_parts: list[str] = []
        if vul_pc and vul_cve:
            sev_select = f", {vul_sev}" if vul_sev else ""
            cve_rows = conn.execute(
                f"SELECT {vul_cve}{sev_select} FROM vulnerabilities WHERE {vul_pc} = ?", (pc,)
            ).fetchall()
            for cr in cve_rows:
                entry = cr[0]
                if vul_sev and len(cr) > 1 and cr[1]:
                    entry += f" ({cr[1]})"
                cve_parts.append(entry)

        doc = f"""PC: {pc}
Department: {m.get(dept_col, 'N/A') if dept_col else 'N/A'}
OS: {m.get(os_col, 'N/A') if os_col else 'N/A'}
CPU: {m.get(cpu_col, 'N/A') if cpu_col else 'N/A'}
RAM: {m.get(ram_col, 'N/A') if ram_col else 'N/A'} GB
Patch status: {m.get(patch_col, 'N/A') if patch_col else 'N/A'}
Vulnerabilities: {m.get(vuln_col, 0) if vuln_col else 0}
Installed software ({len(sw_parts)}): {'; '.join(sw_parts) if sw_parts else 'None recorded'}
CVEs: {'; '.join(cve_parts) if cve_parts else 'None'}"""

        docs.append(doc)

    conn.close()
    return docs


# ---------------------------------------------------------------------------
# Vector store (ChromaDB + sentence-transformers)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building vector store — first run only …")
def get_vector_store():
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings

    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # If empty, build it
    if collection.count() == 0:
        db_path = _resolve_db()
        with st.spinner("Embedding fleet data into vector store …"):
            docs = _load_documents(db_path)
            ids = [f"doc_{i}" for i in range(len(docs))]
            embeddings = embed_model.encode(docs, show_progress_bar=False).tolist()
            # Chroma has a 5,461 doc batch limit — chunk if needed
            batch = 500
            for start in range(0, len(docs), batch):
                collection.add(
                    documents=docs[start:start+batch],
                    embeddings=embeddings[start:start+batch],
                    ids=ids[start:start+batch],
                )

    return collection, embed_model


def retrieve(query: str, collection, embed_model, k: int = TOP_K) -> list[str]:
    q_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    return results["documents"][0] if results["documents"] else []


# ---------------------------------------------------------------------------
# Groq call
# ---------------------------------------------------------------------------

def call_groq(messages: list[dict]) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=groq_messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []   # {"role": ..., "content": ...}

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

# ---------------------------------------------------------------------------
# Load vector store
# ---------------------------------------------------------------------------
with st.spinner("Loading vector store …"):
    try:
        collection, embed_model = get_vector_store()
        st.session_state.vector_store_ready = True
    except Exception as e:
        st.error(f"Failed to build vector store: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar — info + controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 RAG Assistant")
    st.caption(
        f"**Retrieval model:** {EMBED_MODEL}\n\n"
        f"**LLM:** {GROQ_MODEL}\n\n"
        f"**Chunks retrieved per query:** {TOP_K}"
    )
    st.markdown("---")
    st.markdown("**Example questions**")
    example_questions = [
        "Which PCs have critical vulnerabilities?",
        "What is the most common software in Finance?",
        "List PCs that are not fully patched.",
        "Which department has the most CVEs?",
        "What security software is installed across the fleet?",
        "Show PCs with low RAM.",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state._inject_question = q

    st.markdown("---")
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    doc_count = collection.count()
    st.caption(f"Vector store: **{doc_count:,}** documents indexed")

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

# Render history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle injected example question
injected = st.session_state.pop("_inject_question", None)

user_input = st.chat_input("Ask about your IT fleet …") or injected

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Retrieve context
    with st.spinner("Searching fleet data …"):
        chunks = retrieve(user_input, collection, embed_model)
        context = "\n\n---\n\n".join(chunks)
        fleet_summary = load_fleet_summary()

    # Build messages for Groq (include last 6 turns for multi-turn context)
    history_window = st.session_state.chat_history[-6:]
    # Inject fleet summary + retrieved context into the last user message
    augmented_messages = []
    for i, m in enumerate(history_window):
        if i == len(history_window) - 1 and m["role"] == "user":
            augmented_content = (
                f"{fleet_summary}\n\n"
                f"Relevant PC records retrieved for this query:\n\n"
                f"{context}\n\n"
                f"---\n\nUser question: {m['content']}"
            )
            augmented_messages.append({"role": "user", "content": augmented_content})
        else:
            augmented_messages.append(m)

    # Call Gemini
    with st.spinner("Thinking …"):
        try:
            answer = call_groq(augmented_messages)
        except Exception as e:
            answer = f"⚠️ Error calling Groq API: {e}"

    # Show and store assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
