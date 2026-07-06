"""
CVision — AI-Powered Resume Screening Dashboard
================================================
Main Streamlit application for Phase 1 of the CVision pipeline.

Pipeline (Phase 1):
    Google Form → Google Sheet → Knockout Filters → PDF Download
    → pdfplumber Markdown Extraction → BERT Semantic Ranking → Dashboard

Pipeline (Upload mode, no credentials required):
    Upload PDFs → pdfplumber Markdown Extraction → BERT Semantic Ranking → Dashboard

How to run
----------
    streamlit run app.py

    On first run, sentence-transformers will download ~80 MB of model weights
    from HuggingFace Hub. This is cached at ~/.cache/huggingface/ and
    subsequent launches are instant.

Configuration
-------------
Copy .env.example to .env and fill in your values. The app works in full
demo mode without any environment variables (no API keys required for Phase 1).

Dependencies
------------
See requirements.txt. Install with: pip install -r requirements.txt
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Module imports ─────────────────────────────────────────────────────────────
from modules.ingestion import (
    CandidateRecord,
    apply_knockout_filters,
    download_all_resumes,
    fetch_candidates_from_sheet,
    load_mock_candidates,
    DEFAULT_MIN_CGPA,
    DEFAULT_MIN_YEARS_EXP,
)
from modules.parser import extract_markdown_from_pdf, extract_candidate_metadata
from modules.embedding import (
    load_embedding_model,
    rank_resumes_semantic,
    compute_summary_stats,
    extract_skills_from_markdown,
    ACTIVE_BACKEND,
)
from modules.chatbot import build_system_prompt, chat_with_assistant
import os
import pickle
import json

from modules.scheduler_task import get_scheduler, update_scheduler_job, save_scheduler_config
from modules.pipeline_core import run_headless_sheet_pipeline

CACHE_FILE = ".pipeline_cache.pkl"
CONFIG_FILE = "scheduler_config.json"


from modules.email_dispatch import (
    send_filter_rejection_emails,
    send_final_decision_emails,
)

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page configuration — MUST be first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="CVision — AI Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "CVision · Phase 1 · BERT-Powered Resume Screening Pipeline",
    },
)


# ── Cached Resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache the SentenceTransformer model across all Streamlit reruns.

    @st.cache_resource keeps one model instance alive for the lifetime of the
    server process. This avoids the ~2 second reload penalty on every rerun.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer instance.
    """
    return load_embedding_model(model_name)

@st.cache_resource(show_spinner=False)
def _init_scheduler():
    return get_scheduler()

_scheduler = _init_scheduler()


# ── CSS Injection ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    """
    Inject custom CSS for the premium dark glassmorphism theme.

    Uses Streamlit's ``st.markdown(unsafe_allow_html=True)`` to embed a
    ``<style>`` block. All selectors target Streamlit's internal class names
    which are stable across Streamlit 1.35+.
    """
    st.markdown("""
    <style>
    /* ── Google Font ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stTextArea, .stTextInput {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── App background ──────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0d18 0%, #10132a 50%, #0a0d18 100%);
    }

    /* ── PREVENT GREY-OUT during Streamlit reruns ────────────────────────── */
    [data-testid="stAppViewContainer"] > .block-container {
        opacity: 1 !important;
    }
    .stApp > div[data-testid="stAppViewContainer"] {
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    /* Kill Streamlit's built-in skeleton/loading overlay */
    .stApp [data-testid="stAppViewContainer"]::before,
    .stApp [data-testid="stAppViewContainer"]::after {
        display: none !important;
    }
    /* Remove the faded overlay on stale elements */
    .element-container, .stMarkdown, .stAlert, .stButton,
    [data-testid="stVerticalBlock"], [data-testid="column"] {
        opacity: 1 !important;
        transition: none !important;
    }
    /* Ensure sidebar never fades */
    [data-testid="stSidebar"] * {
        opacity: 1 !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    /* Hide the built-in "running" status bar at the top */
    .stStatusWidget, [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ── Custom animated spinner ─────────────────────────────────────────── */
    @keyframes cvision-spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes cvision-pulse {
        0%, 100% { opacity: 0.7; }
        50%      { opacity: 1; }
    }
    /* Override Streamlit's default spinner to use our animation */
    [data-testid="stSpinner"] > div {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }
    [data-testid="stSpinner"] > div > svg,
    [data-testid="stSpinner"] > div > i {
        animation: cvision-spin 1s linear infinite !important;
    }
    [data-testid="stSpinner"] > div > span,
    [data-testid="stSpinner"] > div > p {
        animation: cvision-pulse 1.5s ease-in-out infinite !important;
        color: #a5b4fc !important;
        font-weight: 500 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(10, 13, 24, 0.97) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a5b4fc;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* ── Main header ─────────────────────────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero-title {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin: 0;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        color: #818cf8;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.75rem;
        letter-spacing: 0.5px;
    }

    /* ── Step labels ──────────────────────────────────────────────────────── */
    .step-label {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .step-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
    }

    /* ── KPI cards ────────────────────────────────────────────────────────── */
    .kpi-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        transition: border-color 0.25s ease, transform 0.25s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99,102,241,0.5);
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    .kpi-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .kpi-sub {
        color: #475569;
        font-size: 0.7rem;
        margin-top: 2px;
    }

    /* ── Section headings ─────────────────────────────────────────────────── */
    .section-heading {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 1.5rem 0 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
    }

    /* ── Skill tags ───────────────────────────────────────────────────────── */
    .skill-tag {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 500;
        margin: 2px 3px;
        white-space: nowrap;
        transition: background 0.2s ease;
    }
    .skill-tag:hover {
        background: rgba(99,102,241,0.25);
    }

    /* ── Candidate rank badge ─────────────────────────────────────────────── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* ── Filter status badges ─────────────────────────────────────────────── */
    .badge-pass {
        display: inline-block;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        color: #34d399;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-fail {
        display: inline-block;
        background: rgba(239,68,68,0.12);
        border: 1px solid rgba(239,68,68,0.35);
        color: #f87171;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    /* ── Score bar ────────────────────────────────────────────────────────── */
    .score-bar-track {
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6366f1, #c084fc);
        transition: width 0.6s ease;
    }

    /* ── Dividers ─────────────────────────────────────────────────────────── */
    .glass-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    /* ── Streamlit widget overrides ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        opacity: 0.92 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
        opacity: 1 !important;
    }
    /* Prevent buttons from greying out while Streamlit re-runs */
    .stButton > button:disabled {
        opacity: 0.65 !important;
        cursor: wait !important;
    }

    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    }

    /* ── Expander ─────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* ── Success / Warning / Error boxes ─────────────────────────────────── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Dataframe table overrides ────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    """
    Render the sidebar configuration panel and return a config dict.

    The sidebar collects all user settings so the main content area stays
    clean and focused on results. Wrapping sidebar logic in a function
    keeps the ``main()`` function readable.

    Returns:
        Dict with keys:
        - ``data_source``     : "Upload PDFs" or "Google Sheet"
        - ``sheet_id``        : Google Sheet ID (empty if Upload mode)
        - ``credentials_path``: Path to Service Account JSON (empty if Upload mode)
        - ``min_cgpa``        : Minimum CGPA filter threshold
        - ``min_years_exp``   : Minimum years of experience threshold
        - ``model_name``      : Embedding model identifier (fixed for Phase 1)
    """
    from modules.scheduler_task import _load_scheduler_config
    sched_cfg = _load_scheduler_config() or {}
    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <div style='font-size:2rem;'>🎯</div>
            <div style='background: linear-gradient(135deg, #818cf8, #c084fc);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        font-size:1.3rem; font-weight:800; letter-spacing:-0.5px;'>
                CVision
            </div>
            <div style='color:#475569; font-size:0.7rem; letter-spacing:1.5px;
                        text-transform:uppercase; margin-top:2px;'>
                AI Resume Screening · Phase 1
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Data Source ────────────────────────────────────────────────────────
        st.markdown("### 📥 Data Source")
        data_source = st.radio(
            "Select how to load candidates:",
            options=["📂 Folder Scan", "📁 Upload PDFs", "🔗 Google Sheet"],
            label_visibility="collapsed",
            help=(
                "Folder Scan: auto-scan a local directory for PDF resumes (great for demos).\n\n"
                "Upload PDFs: manually upload resume files — no credentials needed.\n\n"
                "Google Sheet: pull candidates from your Google Form response sheet."
            ),
        )

        sheet_id = ""
        credentials_path = ""
        folder_path = ""

        if data_source == "📂 Folder Scan":
            # Default to the test_cvs folder relative to app.py
            default_folder = str(Path(__file__).parent / "test_cvs")
            folder_path = st.text_input(
                "Resume Folder Path",
                value=default_folder,
                key="folder_path_input",
                placeholder="C:/path/to/resumes",
                help="Absolute path to a folder containing PDF resume files.",
            )
            if folder_path and Path(folder_path).is_dir():
                pdf_count = len(list(Path(folder_path).glob("*.pdf")))
                st.caption(f"📄 Found **{pdf_count}** PDF(s) in folder")
            elif folder_path:
                st.warning("⚠️ Folder not found. Check the path.")

        elif data_source == "🔗 Google Sheet":
            st.text_input(
                "Google Sheet ID",
                value=os.getenv("GOOGLE_SHEET_ID", ""),
                key="sheet_id_input",
                placeholder="1abc...xyz (from Sheet URL)",
                help="The long ID in your Sheet URL between /d/ and /edit",
            )
            sheet_id = st.session_state.get("sheet_id_input", "")

            creds_file = st.file_uploader(
                "Service Account JSON",
                type=["json"],
                key="creds_uploader",
                help="Download from Google Cloud Console → IAM → Service Accounts → Keys",
            )
            if creds_file:
                # Write the uploaded credentials to a temp file so gspread can read it
                tmp_creds = Path(tempfile.gettempdir()) / "cvision_credentials.json"
                tmp_creds.write_bytes(creds_file.read())
                credentials_path = str(tmp_creds)
                st.success("✅ Credentials loaded")
            else:
                credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Knockout Filters ───────────────────────────────────────────────────
        st.markdown("### 🚫 Knockout Filters")
        st.caption("Candidates below these thresholds are auto-rejected.")

        min_cgpa = st.slider(
            "Minimum CGPA (out of 4.0)",
            min_value=0.0,
            max_value=4.0,
            value=float(sched_cfg.get("min_cgpa", DEFAULT_MIN_CGPA)),
            step=0.1,
            format="%.1f",
            key="min_cgpa_slider",
            help="Candidates with CGPA below this value will be filtered out.",
        )

        min_years_exp = st.slider(
            "Minimum Years of Experience",
            min_value=0,
            max_value=15,
            value=int(sched_cfg.get("min_years_exp", DEFAULT_MIN_YEARS_EXP)),
            step=1,
            key="min_years_exp_slider",
            help="Candidates with fewer years will be filtered out.",
        )

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Model Info (dynamically reflects detected backend) ─────────────────
        st.markdown("### 🧠 Embedding Model")
        if ACTIVE_BACKEND == "fastembed":
            backend_label = "fastembed (ONNX)"
            backend_color = "#34d399"
            backend_note  = "ONNX · no PyTorch · Python 3.14 ✅"
        elif ACTIVE_BACKEND == "sentence-transformers":
            backend_label = "sentence-transformers"
            backend_color = "#818cf8"
            backend_note  = "PyTorch · BERT-class embeddings"
        else:
            backend_label = "TF-IDF (fallback)"
            backend_color = "#f59e0b"
            backend_note  = "No BERT backend — install fastembed"
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
                    border-radius:10px; padding:12px; font-size:0.8rem; color:#94a3b8;'>
            <div style='color:#a5b4fc; font-weight:600; margin-bottom:4px;'>
                all-MiniLM-L6-v2
            </div>
            22M params · 384-dim · CPU-fast<br>
            ~85MB (cached after 1st run)<br>
            <span style='color:{backend_color}; font-weight:600;'>
                Backend: {backend_label}
            </span><br>
            <span style='color:#64748b; font-size:0.7rem;'>{backend_note}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Position & Company ─────────────────────────────────────────────────
        st.markdown("### 🏢 Organization")
        position_name = st.text_input(
            "Position Title",
            value=sched_cfg.get("position_name", "the open position"),
            key="position_name_input",
            help="Used in all automated emails.",
        )
        company_name = st.text_input(
            "Company Name",
            value=sched_cfg.get("company_name", "Our Organization"),
            key="company_name_input",
            help="Used in all automated emails.",
        )

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Auto-Email Toggle ──────────────────────────────────────────────────
        st.markdown("### 📧 Auto-Email")
        auto_email = st.toggle(
            "Auto-send filter rejections on completion",
            value=sched_cfg.get("auto_email", False),
            key="auto_email_toggle",
            help="If enabled, candidates who fail hard filters will immediately receive a rejection email when the pipeline finishes.",
        )

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Help link ──────────────────────────────────────────────────────────
        st.caption("📋 See `GOOGLE_FORM_SPEC.md` for form setup & Service Account guide.")

    return {
        "data_source":      data_source,
        "sheet_id":         sheet_id,
        "credentials_path": credentials_path,
        "folder_path":      folder_path,
        "min_cgpa":         min_cgpa,
        "min_years_exp":    float(min_years_exp),
        "model_name":       "all-MiniLM-L6-v2",
        "position_name":    position_name,
        "company_name":     company_name,
        "auto_email":       auto_email,
    }


# ── Result Rendering Functions ─────────────────────────────────────────────────

def _render_kpi_row(stats: dict, n_filtered: int, n_total: int) -> None:
    """
    Render the four KPI metric cards at the top of the results section.

    Args:
        stats: Summary statistics dict from ``compute_summary_stats()``.
        n_filtered: Number of candidates removed by knockout filters.
        n_total: Total candidates (passed + filtered).
    """
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{stats['top_score']:.1f}%",   "Top Fit Score",      "Best matched candidate"),
        (c2, f"{stats['avg_score']:.1f}%",   "Average Score",       f"Std dev: {stats['score_std']:.1f}%"),
        (c3, str(stats["n_candidates"]),      "Candidates Ranked",  f"{n_filtered} filtered out"),
        (c4, str(stats["above_70"]),          "Strong Matches",      "Score ≥ 70%"),
    ]
    for col, value, label, sub in cards:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_score_chart(results_df: pd.DataFrame) -> None:
    """
    Render a Plotly horizontal bar chart of fit scores with gradient fill.

    Plotly is used instead of Streamlit's native chart for precise control
    over colors, hover tooltips, axis labels, and background theming.

    Args:
        results_df: Ranked DataFrame from ``rank_resumes_semantic()``.
    """
    if results_df.empty:
        return

    # Reverse for bottom-to-top display (highest score at top)
    df_plot = results_df.sort_values("Fit Score (%)", ascending=True)

    # Use candidate name if available; fall back to filename
    if "Candidate Name" in df_plot.columns:
        labels = df_plot["Candidate Name"].tolist()
    else:
        labels = [Path(f).stem for f in df_plot["Filename"].tolist()]

    scores = df_plot["Fit Score (%)"].tolist()

    # Color-code bars by score tier
    colors = [
        "#10b981" if s >= 70 else "#6366f1" if s >= 50 else "#475569"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
        hovertemplate="<b>%{y}</b><br>Fit Score: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(
            title="Fit Score (%)",
            range=[0, 105],
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#64748b"),
            title_font=dict(color="#64748b"),
        ),
        yaxis=dict(
            tickfont=dict(color="#e2e8f0", size=12),
        ),
        margin=dict(l=0, r=60, t=10, b=10),
        height=max(300, len(labels) * 48),
        showlegend=False,
    )

    # Add a vertical reference line at 70% (strong match threshold)
    fig.add_vline(
        x=70,
        line_dash="dot",
        line_color="rgba(16,185,129,0.4)",
        annotation_text="Strong (70%)",
        annotation_position="top right",
        annotation_font=dict(color="#34d399", size=11),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_ranked_table(results_df: pd.DataFrame) -> None:
    """
    Display the ranked results table with colour-coded scores.

    Uses ``st.dataframe`` with pandas Styler for per-cell formatting.
    The score column is highlighted on a red→green gradient.

    Args:
        results_df: Ranked DataFrame from ``rank_resumes_semantic()``.
    """
    if results_df.empty:
        return

    def _color_score(val: float) -> str:
        """Return a CSS color string for a given score value."""
        if val >= 70:
            return "color: #34d399; font-weight: 700;"
        elif val >= 50:
            return "color: #818cf8; font-weight: 600;"
        else:
            return "color: #64748b;"

    def _style_rank(val: int) -> str:
        """Bold rank 1."""
        return "font-weight: 700; color: #f472b6;" if val == 1 else "color: #94a3b8;"

    # Select display columns (only those that exist in the DataFrame)
    display_cols = [
        c for c in [
            "Rank", "Candidate Name", "Filename",
            "Fit Score (%)", "CGPA", "Degree", "Top Skills"
        ] if c in results_df.columns
    ]

    display_df = results_df[display_cols].copy()
    display_df["Fit Score (%)"] = display_df["Fit Score (%)"].round(2)

    styled = (
        display_df.style
        .map(_color_score, subset=["Fit Score (%)"])
        .map(_style_rank, subset=["Rank"])
        .format({"Fit Score (%)": "{:.2f}%"})
        .set_properties(**{
            "background-color": "rgba(10,13,24,0.9)",
            "color": "#e2e8f0",
        })
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_candidate_details(
    results_df: pd.DataFrame,
    candidates: list[CandidateRecord],
) -> None:
    """
    Render an expandable detail card for each ranked candidate.

    Shows: rank badge, fit score + CGPA badge, skills tags, metadata fields,
    and clickable profile links (LinkedIn, GitHub, Portfolio).

    Args:
        results_df: Ranked DataFrame from ``rank_resumes_semantic()``.
        candidates: List of ``CandidateRecord`` instances for metadata lookup.
    """
    # Build name→record and filename→record lookups
    record_by_name: dict[str, CandidateRecord] = {c.name: c for c in candidates}
    record_by_file: dict[str, CandidateRecord] = {}
    for c in candidates:
        if c.local_resume_path:
            record_by_file[c.local_resume_path.name] = c
        record_by_file[c.name] = c

    st.markdown("<div class='section-heading'>👤 Candidate Details</div>", unsafe_allow_html=True)

    for _, row in results_df.iterrows():
        rank  = int(row["Rank"])
        score = float(row["Fit Score (%)"])
        cand_name = row.get("Candidate Name", Path(row["Filename"]).stem)
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"

        with st.expander(f"{emoji} Rank #{rank} — {cand_name} · {score:.1f}%", expanded=(rank == 1)):
            col_left, col_right = st.columns([3, 2])

            # Resolve record once for use in both columns
            record = record_by_name.get(cand_name) or record_by_file.get(row["Filename"])

            # ── Left column: score bar + CGPA badge + skills ──────────────────
            with col_left:
                bar_width = min(int(score), 100)

                # CGPA badge shown inline next to the score label
                cgpa_html = ""
                if record and record.cgpa != -1.0:
                    cgpa_color = (
                        "#34d399" if record.cgpa >= 3.5
                        else "#818cf8" if record.cgpa >= 3.0
                        else "#f59e0b"
                    )
                    cgpa_html = (
                        f"<span style='background:rgba(99,102,241,0.12); "
                        f"border:1px solid rgba(99,102,241,0.3); color:{cgpa_color}; "
                        f"border-radius:20px; padding:2px 10px; font-size:0.72rem; "
                        f"font-weight:700; margin-left:8px;'>CGPA&nbsp;{record.cgpa:.2f}</span>"
                    )

                st.markdown(f"""
                <div style='margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:6px;'>
                        <span style='color:#94a3b8; font-size:0.8rem;'>Semantic Fit Score{cgpa_html}</span>
                        <span style='color:#818cf8; font-weight:700; font-size:1.1rem;'>{score:.1f}%</span>
                    </div>
                    <div class='score-bar-track'>
                        <div class='score-bar-fill' style='width:{bar_width}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Skills tags
                skills_str = row.get("Top Skills", "")
                if skills_str and skills_str != "—":
                    skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]
                    tags_html = "".join(
                        f"<span class='skill-tag'>{s}</span>" for s in skills_list
                    )
                    st.markdown(
                        f"<div style='margin-bottom:10px;'><strong style='color:#94a3b8;"
                        f"font-size:0.75rem;'>🔧 TOP SKILLS</strong><br>{tags_html}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span style='color:#475569; font-size:0.8rem;'>No skills detected in resume</span>",
                        unsafe_allow_html=True,
                    )

            with col_right:
                if record:
                    meta_items = []
                    if record.email:
                        meta_items.append(("📧 Email", record.email))
                    if record.cgpa != -1.0:
                        meta_items.append(("🎓 CGPA", f"{record.cgpa:.2f} / 4.0"))
                    if record.degree:
                        meta_items.append(("🏛 Degree", record.degree.title()))
                    if record.major:
                        meta_items.append(("📚 Major", record.major))
                    if record.position_applied:
                        meta_items.append(("🎯 Applying For", record.position_applied))
                    if record.phone:
                        meta_items.append(("📱 Phone", record.phone))

                    if meta_items:
                        for label, value in meta_items:
                            st.markdown(
                                f"<div style='margin-bottom:6px;'>"
                                f"<span style='color:#64748b;font-size:0.72rem;'>{label}</span><br>"
                                f"<span style='color:#e2e8f0;font-size:0.85rem;font-weight:500;'>{value}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<span style='color:#475569; font-size:0.8rem;'>"
                            "No structured metadata found in CV text.</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<span style='color:#475569; font-size:0.8rem;'>Metadata not available.</span>",
                        unsafe_allow_html=True,
                    )

def _render_filtered_candidates(filtered: list[CandidateRecord]) -> None:
    """
    Display a collapsed panel listing candidates rejected by knockout filters.

    Showing filtered candidates (with reasons) is important for HR auditability:
    recruiters need to know *why* each application was not advanced, and have the
    option to manually override the decision.

    Args:
        filtered: List of ``CandidateRecord`` instances with ``passed_filter=False``.
    """
    if not filtered:
        return

    st.markdown("<div class='section-heading'>🚫 Filtered Out by Knockout Rules</div>", unsafe_allow_html=True)

    with st.expander(f"Show {len(filtered)} screened-out candidate(s)", expanded=False):
        rows = []
        for c in filtered:
            rows.append({
                "Name":   c.name,
                "Email":  c.email,
                "CGPA":   f"{c.cgpa:.2f}" if c.cgpa != -1.0 else "N/A",
                "Degree": c.degree.title() if c.degree else "N/A",
                "Reason": c.filter_reason,
            })

        filtered_df = pd.DataFrame(rows)
        st.dataframe(
            filtered_df.style.map(
                lambda _: "color: #f87171;", subset=["Reason"]
            ),
            use_container_width=True,
            hide_index=True,
        )

# ── AI HR Chatbot ──────────────────────────────────────────────────────────────

def _render_chatbot(
    results_df: pd.DataFrame,
    candidates: list[CandidateRecord],
    job_description: str,
    filtered: list[CandidateRecord],
) -> None:
    """
    Render the AI HR Chatbot panel using Gemini.

    Builds a system prompt from pipeline results and maintains conversation
    history in session state. Provides a chat interface for the recruiter
    to ask questions about candidates.
    """
    st.markdown(
        "<div class='section-heading'>🤖 AI HR Assistant</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask me anything about the candidates — comparisons, skill gaps, "
        "recommendations, or deep dives into specific CVs."
    )

    # Build system prompt if not already cached (or if results changed)
    results_hash = hash(results_df.to_json())
    if (
        "chatbot_system_prompt" not in st.session_state
        or st.session_state.get("chatbot_results_hash") != results_hash
    ):
        st.session_state["chatbot_system_prompt"] = build_system_prompt(
            results_df=results_df,
            candidates=candidates,
            job_description=job_description,
            filtered=filtered,
        )
        st.session_state["chatbot_results_hash"] = results_hash
        st.session_state["chat_history"] = []

    # Display existing chat messages
    for msg in st.session_state.get("chat_history", []):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["parts"][0])

    # Chat input
    user_input = st.chat_input(
        "Ask about candidates... (e.g., 'Who is the best fit for this role?')",
        key="chatbot_input",
    )

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get LLM response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat_with_assistant(
                        user_message=user_input,
                        chat_history=st.session_state.get("chat_history", []),
                        system_prompt=st.session_state["chatbot_system_prompt"],
                    )
                    st.markdown(response)

                    # Update chat history
                    st.session_state.setdefault("chat_history", []).extend([
                        {"role": "user", "parts": [user_input]},
                        {"role": "model", "parts": [response]},
                    ])
                except Exception as exc:
                    st.error(f"❌ Chatbot error: {exc}")
                    logger.error("Chatbot error: %s", exc)


# ── Candidate Selection & Email Dispatch ──────────────────────────────────────

def _render_candidate_selection(
    results_df: pd.DataFrame,
    candidates: list[CandidateRecord],
    filtered: list[CandidateRecord],
) -> None:
    """
    Render candidate selection checkboxes and email dispatch controls.

    Allows the recruiter to select candidates for the next round, then
    triggers automated acceptance/rejection emails.
    """
    st.markdown(
        "<div class='section-heading'>✅ Finalize Candidate Selection</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Select candidates to advance to the next round. "
        "Unselected candidates will receive a rejection email."
    )

    # Build the candidate list from results_df
    record_by_name = {c.name: c for c in candidates}
    selected_names = []

    # Two-column layout: selections on left, summary on right
    sel_col, sum_col = st.columns([3, 2])

    with sel_col:
        for _, row in results_df.iterrows():
            cand_name = row.get("Candidate Name", row.get("Filename", "Unknown"))
            score = row.get("Fit Score (%)", 0)
            cgpa = row.get("CGPA", "N/A")

            # Medal emoji based on rank
            rank = row.get("Rank", 0)
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📋"

            checked = st.checkbox(
                f"{medal} {cand_name} — {score:.1f}% fit | CGPA: {cgpa}",
                key=f"select_{cand_name}",
                value=rank <= 3,  # Pre-select top 3
            )
            if checked:
                selected_names.append(cand_name)

    with sum_col:
        total = len(results_df)
        n_selected = len(selected_names)
        n_rejected = total - n_selected

        st.markdown(f"""
        <div style='background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2);
                    border-radius: 12px; padding: 16px; margin-bottom: 12px;'>
            <div style='color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;'>
                Selection Summary
            </div>
            <div style='margin-top: 8px;'>
                <span style='color: #34d399; font-size: 1.2rem; font-weight: 700;'>{n_selected}</span>
                <span style='color: #94a3b8; font-size: 0.85rem;'> selected for next round</span>
            </div>
            <div style='margin-top: 4px;'>
                <span style='color: #f87171; font-size: 1.2rem; font-weight: 700;'>{n_rejected}</span>
                <span style='color: #94a3b8; font-size: 0.85rem;'> will receive rejection</span>
            </div>
            <div style='margin-top: 4px;'>
                <span style='color: #fbbf24; font-size: 1.2rem; font-weight: 700;'>{len(filtered)}</span>
                <span style='color: #94a3b8; font-size: 0.85rem;'> filtered out (pre-screening)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Store selection in session state
    st.session_state["selected_names"] = selected_names

    # ── Email Dispatch Controls ───────────────────────────────────────────────
    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

    email_col1, email_col2 = st.columns(2)

    position_name = st.session_state.get("position_name_input", "the open position")
    company_name = st.session_state.get("company_name_input", "Our Organization")

    with email_col1:
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.8rem;'>📌 Position: "
            f"<strong style='color:#e2e8f0;'>{position_name}</strong></div>",
            unsafe_allow_html=True,
        )
    with email_col2:
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.8rem;'>🏢 Company: "
            f"<strong style='color:#e2e8f0;'>{company_name}</strong></div>",
            unsafe_allow_html=True,
        )
    st.caption("Edit these in the sidebar under 🏢 Organization.")

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        preview_btn = st.button(
            "👁️ Preview Emails (Dry Run)",
            key="preview_emails_btn",
            use_container_width=True,
        )

    with btn_col2:
        send_btn = st.button(
            "📧 Send Decision Emails",
            type="primary",
            key="send_emails_btn",
            use_container_width=True,
        )

    with btn_col3:
        send_filter_btn = st.button(
            "🚫 Send Filter Rejections",
            key="send_filter_emails_btn",
            use_container_width=True,
        )

    if preview_btn or send_btn:
        dry_run = preview_btn
        mode_label = "PREVIEW" if dry_run else "SENDING"

        # Separate selected and rejected
        selected_records = [c for c in candidates if c.name in selected_names]
        rejected_records = [c for c in candidates if c.name not in selected_names]

        with st.spinner(f"📧 {mode_label} decision emails..."):
            results = send_final_decision_emails(
                selected=selected_records,
                rejected=rejected_records,
                position=position_name,
                company_name=company_name,
                dry_run=dry_run,
            )

        if dry_run:
            st.info(
                f"👁️ **Preview complete** — {results['accepted_sent']} acceptance + "
                f"{results['rejected_sent']} rejection emails would be sent. "
                f"({results['skipped']} skipped — no email on file)"
            )
        else:
            st.success(
                f"✅ **Emails sent!** {results['accepted_sent']} acceptance + "
                f"{results['rejected_sent']} rejection emails dispatched."
            )

    if send_filter_btn:
        with st.spinner("📧 Sending filter rejection emails..."):
            results = send_filter_rejection_emails(
                filtered_candidates=filtered,
                position=position_name,
                company_name=company_name,
                dry_run=False,
            )
        st.success(
            f"✅ Filter rejection emails: {results['sent']} sent, "
            f"{results['failed']} failed, {results['skipped']} skipped."
        )


# ── Pipeline Scheduling ───────────────────────────────────────────────────────

def _render_scheduling() -> None:
    """
    Render the pipeline scheduling controls.

    Allows the user to set a specific time for the pipeline to run
    automatically, or trigger it manually.
    """
    st.markdown(
        "<div class='section-heading'>⏰ Pipeline Scheduling</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Schedule the pipeline to run automatically at a set time, "
        "or trigger it manually with the button above."
    )

    sched_col1, sched_col2, sched_col3 = st.columns([2, 2, 1])

    # Load defaults from config if present
    from modules.scheduler_task import _load_scheduler_config
    from datetime import datetime
    sched_cfg = _load_scheduler_config() or {}
    default_enabled = sched_cfg.get("schedule_enabled", False)
    default_time_str = sched_cfg.get("schedule_time", "")
    default_time = None
    if default_time_str:
        try:
            default_time = datetime.strptime(default_time_str, "%H:%M").time()
        except:
            pass

    with sched_col1:
        schedule_time = st.time_input(
            "Scheduled Run Time",
            value=default_time,
            key="schedule_time",
            help="The pipeline will automatically run at this time daily.",
        )

    with sched_col2:
        schedule_enabled = st.toggle(
            "Enable Daily Schedule",
            value=default_enabled,
            key="schedule_enabled",
            help="Turn on to automatically run the pipeline at the scheduled time.",
        )

    with sched_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if schedule_enabled and schedule_time:
            st.markdown(
                f"<div style='background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); "
                f"border-radius: 8px; padding: 8px 12px; text-align: center;'>"
                f"<span style='color: #34d399; font-weight: 600;'>🟢 Active</span><br>"
                f"<span style='color: #94a3b8; font-size: 0.75rem;'>Next run: {schedule_time.strftime('%I:%M %p')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background: rgba(248, 113, 113, 0.08); border: 1px solid rgba(248, 113, 113, 0.2); "
                "border-radius: 8px; padding: 8px 12px; text-align: center;'>"
                "<span style='color: #f87171; font-weight: 600;'>🔴 Inactive</span>"
                "</div>",
                unsafe_allow_html=True,
            )

    # Hook up scheduler
    time_str = schedule_time.strftime("%H:%M") if schedule_time else ""
    update_scheduler_job(_scheduler, time_str, schedule_enabled)

    st.markdown("### 💾 Configuration")
    st.caption("Save your current Job Description and Filters so the automated scheduler knows what to run.")
    
    col_save, col_reset = st.columns([3, 1])
    with col_save:
        if st.button("💾 Save Current Configuration for Scheduler", use_container_width=True):
            jd_val = st.session_state.get("jd_input", "")
            if not jd_val.strip():
                st.error("⚠️ Please paste a Job Description in the main panel before saving config.")
            else:
                config = {
                    "job_description": jd_val.strip(),
                    "sheet_id": st.session_state.get("sheet_id_input", st.session_state.get("sheet_id", "")),
                    "credentials_path": "credentials.json",
                    "min_cgpa": float(st.session_state.get("min_cgpa_slider", DEFAULT_MIN_CGPA)),
                    "min_years_exp": float(st.session_state.get("min_years_exp_slider", DEFAULT_MIN_YEARS_EXP)),
                    "model_name": "all-MiniLM-L6-v2",
                    "auto_email": st.session_state.get("auto_email_toggle", False),
                    "position_name": st.session_state.get("position_name_input", "the open position"),
                    "company_name": st.session_state.get("company_name_input", "Our Organization"),
                    "schedule_time": schedule_time.strftime("%H:%M") if schedule_time else "",
                    "schedule_enabled": schedule_enabled,
                }
                config_json = save_scheduler_config(config)
                st.success("✅ Configuration saved! The scheduler will use these settings.")
                with st.expander("📋 For Render hosting: copy this to your Environment Variables"):
                    st.caption("On Render Dashboard → Environment → Add variable:")
                    st.code(f"SCHEDULER_CONFIG={config_json}", language="text")
    
    with col_reset:
        if st.button("🔄 Reset defaults", use_container_width=True):
            if "SCHEDULER_CONFIG" in os.environ:
                del os.environ["SCHEDULER_CONFIG"]
            from modules.scheduler_task import CONFIG_FILE
            if Path(CONFIG_FILE).exists():
                try:
                    Path(CONFIG_FILE).unlink()
                except:
                    pass
            st.success("✅ Defaults restored! (If on Render, please delete the SCHEDULER_CONFIG environment variable in your dashboard as well)")
            st.rerun()

    # Store schedule state
    if schedule_enabled and schedule_time:
        st.session_state["scheduled_time"] = schedule_time
        st.session_state["schedule_active"] = True
    else:
        st.session_state["schedule_active"] = False


# ── Pipeline Orchestration ─────────────────────────────────────────────────────


def _run_pipeline_folder_mode(
    jd: str,
    folder_path: str,
    model,
) -> tuple[pd.DataFrame, dict, list[CandidateRecord], list[CandidateRecord]]:
    """
    Execute the pipeline in Folder Scan Mode.

    Scans a local directory for PDF files, extracts Markdown from each,
    and runs BERT semantic ranking. No credentials or uploads required —
    ideal for demos and automated pipeline runs.

    Args:
        jd: Job description text.
        folder_path: Absolute path to the folder containing PDF resumes.
        model: Loaded SentenceTransformer / fastembed model.

    Returns:
        Tuple of (results_df, stats, passed_candidates, filtered_candidates).
        filtered_candidates is always empty in folder mode.
    """
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        st.error(f"No PDF files found in `{folder_path}`. Check the folder path.")
        st.stop()

    st.info(f"📂 Found **{len(pdf_files)}** PDF(s) in `{folder.name}/`")

    filenames: list[str] = []
    resume_markdowns: list[str] = []
    skipped: list[str] = []

    progress = st.progress(0, text="Extracting text from PDFs...")

    for i, pdf_path in enumerate(pdf_files):
        progress.progress(
            (i + 1) / len(pdf_files),
            text=f"Parsing: {pdf_path.name} ({i+1}/{len(pdf_files)})",
        )
        markdown = extract_markdown_from_pdf(pdf_path)

        if markdown and markdown.strip():
            filenames.append(pdf_path.name)
            resume_markdowns.append(markdown)
        else:
            skipped.append(pdf_path.name)
            st.warning(f"⚠️ Could not extract text from **{pdf_path.name}** — skipped.")

    progress.empty()

    if skipped:
        with st.expander(f"⚠️ {len(skipped)} file(s) skipped (empty extraction)", expanded=False):
            for name in skipped:
                st.markdown(f"- `{name}`")

    if not filenames:
        st.error("No readable PDFs found. Please check your files and try again.")
        st.stop()

    # Build CandidateRecord objects enriched with auto-extracted metadata
    candidates = []
    candidate_metadata = []
    for fn, md in zip(filenames, resume_markdowns):
        meta = extract_candidate_metadata(md)   # email, phone, linkedin, github, portfolio, cgpa
        display_name = Path(fn).stem.replace("_", " ").replace("-", " ").title()
        rec = CandidateRecord(
            name=display_name,
            email=meta.get("email", ""),
            phone=meta.get("phone", ""),
            linkedin_url=meta.get("linkedin", ""),
            cgpa=meta.get("cgpa", -1.0),           # extracted from CV text
            resume_markdown=md,
        )
        # Stash github + portfolio in job_title / cover_note (spare string fields)
        # so the renderer can retrieve them without a schema change.
        rec.job_title  = meta.get("github", "")
        rec.cover_note = meta.get("portfolio", "")
        candidates.append(rec)
        candidate_metadata.append({
            "name":      display_name,
            "email":     rec.email,
            "cgpa":      rec.cgpa,
            "years_exp": rec.years_exp,  # -1.0 = not found
            "degree":    rec.degree,
        })

    with st.spinner("🧠 Running BERT semantic ranking..."):
        results_df = rank_resumes_semantic(
            job_description=jd,
            filenames=filenames,
            resume_markdowns=resume_markdowns,
            model=model,
            candidate_metadata=candidate_metadata,
        )

    stats = compute_summary_stats(results_df)
    return results_df, stats, candidates, []


def _run_pipeline_upload_mode(
    jd: str,
    uploaded_files: list,
    model,
) -> tuple[pd.DataFrame, dict, list[CandidateRecord], list[CandidateRecord]]:
    """
    Execute the pipeline in Upload Mode (no Google Sheet, no filters).

    Steps:
    1. Extract Markdown from each uploaded PDF using pdfplumber.
    2. Run BERT semantic ranking against the Job Description.
    3. Return results, stats, and minimal CandidateRecord list.

    Args:
        jd: Job description text.
        uploaded_files: List of Streamlit UploadedFile objects.
        model: Loaded SentenceTransformer model.

    Returns:
        Tuple of (results_df, stats, passed_candidates, filtered_candidates).
        In Upload mode, filtered_candidates is always empty.
    """
    filenames: list[str] = []
    resume_markdowns: list[str] = []
    skipped: list[str] = []

    progress = st.progress(0, text="Extracting text from PDFs...")

    for i, pdf_file in enumerate(uploaded_files):
        progress.progress(
            (i + 1) / len(uploaded_files),
            text=f"Parsing: {pdf_file.name} ({i+1}/{len(uploaded_files)})",
        )
        # Read the uploaded file into BytesIO so pdfplumber can open it
        file_bytes = io.BytesIO(pdf_file.read())
        markdown = extract_markdown_from_pdf(file_bytes)

        if markdown and markdown.strip():
            filenames.append(pdf_file.name)
            resume_markdowns.append(markdown)
        else:
            skipped.append(pdf_file.name)
            st.warning(f"⚠️ Could not extract text from **{pdf_file.name}** — skipped.")

    progress.empty()

    if not filenames:
        st.error("No readable PDFs found. Please check your files and try again.")
        st.stop()

    # Build CandidateRecord objects enriched with auto-extracted metadata
    candidates = []
    candidate_metadata = []
    for fn, md in zip(filenames, resume_markdowns):
        meta = extract_candidate_metadata(md)   # email, phone, linkedin, github, portfolio, cgpa
        display_name = Path(fn).stem.replace("_", " ").replace("-", " ").title()
        rec = CandidateRecord(
            name=display_name,
            email=meta.get("email", ""),
            phone=meta.get("phone", ""),
            linkedin_url=meta.get("linkedin", ""),
            cgpa=meta.get("cgpa", -1.0),           # extracted from CV text
            resume_markdown=md,
        )
        # Stash github + portfolio in job_title / cover_note (spare string fields)
        rec.job_title  = meta.get("github", "")
        rec.cover_note = meta.get("portfolio", "")
        candidates.append(rec)
        candidate_metadata.append({
            "name":      display_name,
            "email":     rec.email,
            "cgpa":      rec.cgpa,
            "years_exp": rec.years_exp,  # -1.0 = not found
            "degree":    rec.degree,
        })

    with st.spinner("🧠 Running BERT semantic ranking..."):
        results_df = rank_resumes_semantic(
            job_description=jd,
            filenames=filenames,
            resume_markdowns=resume_markdowns,
            model=model,
            candidate_metadata=candidate_metadata,
        )

    stats = compute_summary_stats(results_df)
    return results_df, stats, candidates, []


def _run_pipeline_sheet_mode(
    jd: str,
    sheet_id: str,
    credentials_path: str,
    min_cgpa: float,
    min_years_exp: float,
    model,
) -> tuple[pd.DataFrame, dict, list, list]:
    with st.status("🚀 Running Automated Pipeline...", expanded=True) as status:
        try:
            def update_status(msg: str):
                status.update(label=msg, state="running")
            
            res = run_headless_sheet_pipeline(
                jd, sheet_id, credentials_path, min_cgpa, min_years_exp, model,
                progress_callback=update_status
            )
            status.update(label="✅ Pipeline Completed Successfully!", state="complete", expanded=False)
            return res
        except Exception as e:
            status.update(label="❌ Pipeline Failed", state="error", expanded=True)
            st.error(f"Pipeline Error: {e}")
            st.stop()
            
def _old_sheet_mode_discarded():
    """
    Execute the full pipeline in Google Sheet Mode.

    Steps:
    1. Fetch candidate rows from the linked Google Sheet via gspread.
    2. Apply knockout filters (CGPA, years of experience, degree).
    3. Download resume PDFs from Google Drive for passing candidates.
    4. Extract Markdown from each PDF using pdfplumber.
    5. Run BERT semantic ranking.
    6. Return results enriched with candidate metadata.

    Args:
        jd: Job description text.
        sheet_id: Google Sheet ID string.
        credentials_path: Path to Service Account JSON key.
        min_cgpa: Minimum CGPA threshold for the filter.
        min_years_exp: Minimum years of experience threshold.
        model: Loaded SentenceTransformer model.

    Returns:
        Tuple of (results_df, stats, passed_candidates, filtered_candidates).
    """
    # ── Step 1: Fetch from Google Sheet ───────────────────────────────────────
    with st.spinner("📊 Fetching candidates from Google Sheet..."):
        try:
            all_candidates = fetch_candidates_from_sheet(sheet_id, credentials_path)
        except Exception as exc:
            st.error(f"**Could not connect to Google Sheet.**\n\n{exc}")
            st.info("Tip: Check your Sheet ID and ensure the Service Account has Viewer access.")
            st.stop()

    if not all_candidates:
        st.warning("No candidate submissions found in the Google Sheet.")
        st.stop()

    st.info(f"📋 Fetched **{len(all_candidates)}** submission(s) from Google Sheet.")

    # ── Step 2: Apply knockout filters ────────────────────────────────────────
    with st.spinner("🚫 Applying knockout filters..."):
        apply_knockout_filters(all_candidates, min_cgpa=min_cgpa, min_years_exp=min_years_exp)

    passed    = [c for c in all_candidates if c.passed_filter]
    filtered  = [c for c in all_candidates if not c.passed_filter]

    col_a, col_b = st.columns(2)
    col_a.metric("✅ Passed Filters", len(passed))
    col_b.metric("🚫 Filtered Out",   len(filtered))

    if not passed:
        st.error("All candidates were filtered out. Try lowering the knockout thresholds.")
        # Still show the filtered table so user can diagnose the issue
        _render_filtered_candidates(filtered)
        st.stop()

    # ── Step 3: Download PDFs from Google Drive ───────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="cvision_"))
    with st.spinner("📥 Downloading resumes from Google Drive..."):
        download_all_resumes(
            candidates=passed,
            dest_dir=tmp_dir,
            credentials_path=credentials_path,
            delay_seconds=0.3,
        )

    # Report download success/failure
    downloaded  = [c for c in passed if c.local_resume_path]
    no_download = [c for c in passed if not c.local_resume_path]

    if no_download:
        st.warning(
            f"⚠️ Could not download {len(no_download)} resume(s): "
            + ", ".join(c.name for c in no_download)
        )

    if not downloaded:
        st.error(
            "No resumes could be downloaded. "
            "Check that the Drive folder is shared with the Service Account."
        )
        st.stop()

    # ── Step 4: Extract Markdown from each PDF ────────────────────────────────
    progress = st.progress(0, text="Extracting text from resumes...")
    filenames:        list[str] = []
    resume_markdowns: list[str] = []
    candidate_meta:   list[dict] = []

    for i, candidate in enumerate(downloaded):
        progress.progress(
            (i + 1) / len(downloaded),
            text=f"Parsing: {candidate.name} ({i+1}/{len(downloaded)})",
        )
        markdown = extract_markdown_from_pdf(candidate.local_resume_path)
        candidate.resume_markdown = markdown   # Store on record for future use

        if markdown.strip():
            filenames.append(candidate.local_resume_path.name)
            resume_markdowns.append(markdown)
            candidate_meta.append({
                "name":      candidate.name,
                "email":     candidate.email,
                "cgpa":      candidate.cgpa,
                "years_exp": candidate.years_exp,
                "degree":    candidate.degree,
            })
        else:
            st.warning(f"⚠️ Empty extraction for: {candidate.name} — skipped from ranking.")

    progress.empty()

    if not filenames:
        st.error("Text extraction failed for all resumes.")
        st.stop()

    # ── Step 5: BERT semantic ranking ─────────────────────────────────────────
    with st.spinner("🧠 Running BERT semantic ranking..."):
        results_df = rank_resumes_semantic(
            job_description=jd,
            filenames=filenames,
            resume_markdowns=resume_markdowns,
            model=model,
            candidate_metadata=candidate_meta,
        )

    stats = compute_summary_stats(results_df)
    return results_df, stats, downloaded, filtered


# ── Main Application ───────────────────────────────────────────────────────────

def main() -> None:
    """
    Main Streamlit application entry point.

    Orchestrates the full UI: sidebar config → job description → candidate source
    → pipeline execution → results dashboard. All state is managed via
    ``st.session_state`` so results persist across Streamlit reruns.
    """
    if "results_df" not in st.session_state and Path(CACHE_FILE).exists():
        try:
            import pickle
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            st.session_state["results_df"] = cache_data["results_df"]
            st.session_state["stats"] = cache_data["stats"]
            st.session_state["candidates"] = cache_data["candidates"]
            st.session_state["filtered"] = cache_data["filtered"]
            st.session_state["job_description"] = cache_data["job_description"]
            run_time = cache_data.get("last_run", "").split(".")[0].replace("T", " ")
            st.toast(f"Loaded background run from {run_time}", icon="🔄")
        except Exception:
            pass

    _inject_css()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    config = _render_sidebar()

    # ── Hero Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-container'>
        <h1 class='hero-title'>CVision</h1>
        <div class='hero-subtitle'>AI-Powered Resume Screening · Phase 1 Pipeline</div>
        <span class='hero-badge'>BERT Semantic Ranking · pdfplumber · Google Forms</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

    # ── Step 1: Job Description ────────────────────────────────────────────────
    st.markdown("""
    <div class='step-label'>
        <span class='step-number'>1</span>
        <span class='step-title'>Job Description</span>
    </div>
    """, unsafe_allow_html=True)

    # Load JD default: saved config > job_description.txt > empty
    from modules.scheduler_task import _load_scheduler_config as _load_cfg
    _saved_cfg = _load_cfg() or {}
    _jd_default = _saved_cfg.get("job_description", "")

    if not _jd_default:
        _jd_file = Path(__file__).parent / "job_description.txt"
        if _jd_file.exists():
            try:
                _jd_default = _jd_file.read_text(encoding="utf-8").strip()
                st.caption(
                    f"📄 Pre-loaded from `job_description.txt` — edit below to override, "
                    f"or update the file directly for permanent changes."
                )
            except Exception:
                pass

    job_description = st.text_area(
        "Job Description",
        value=_jd_default,
        height=220,
        placeholder=(
            "Paste the full job description here.\n\n"
            "Include required skills, experience, responsibilities, and qualifications. "
            "The more detail you provide, the more accurate the semantic ranking will be."
        ),
        label_visibility="collapsed",
        key="jd_input",
    )

    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

    # ── Step 2: Candidate Source ────────────────────────────────────────────────
    st.markdown("""
    <div class='step-label'>
        <span class='step-number'>2</span>
        <span class='step-title'>Candidate Source</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = None

    if config["data_source"] == "📂 Folder Scan":
        folder = Path(config["folder_path"]) if config["folder_path"] else Path(__file__).parent / "test_cvs"
        if folder.is_dir():
            pdf_list = sorted(folder.glob("*.pdf"))
            if pdf_list:
                st.markdown(
                    f"<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;'>"
                    f"📂 Scanning <code style='color:#818cf8;'>{folder.name}/</code> — "
                    f"<strong style='color:#e2e8f0;'>{len(pdf_list)} PDF(s)</strong> found:</div>",
                    unsafe_allow_html=True,
                )
                for p in pdf_list:
                    st.markdown(
                        f"<div style='color:#64748b; font-size:0.78rem; padding-left:1rem;'>"
                        f"📄 {p.name}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning(f"No PDF files found in `{folder}`")
        else:
            st.warning("Folder path is not set or does not exist. Check the sidebar.")

    elif config["data_source"] == "📁 Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload PDF Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF resume files to analyze.",
            label_visibility="collapsed",
            key="pdf_uploader",
        )
        if uploaded_files:
            st.caption(f"📄 {len(uploaded_files)} file(s) selected")

    else:  # Google Sheet mode
        if not config["sheet_id"]:
            st.info(
                "🔗 Enter your **Google Sheet ID** in the sidebar and upload your "
                "**Service Account JSON** credentials to load candidates automatically."
            )
        else:
            st.success(f"🔗 Connected to Sheet: `{config['sheet_id'][:20]}...`")
        
        st.session_state["sheet_id"] = config["sheet_id"]
        st.session_state["min_cgpa"] = config["min_cgpa"]
        st.session_state["min_years_exp"] = config["min_years_exp"]

    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

    # ── Step 3: Analyze Button ─────────────────────────────────────────────────
    st.markdown("""
    <div class='step-label'>
        <span class='step-number'>3</span>
        <span class='step-title'>Run Pipeline</span>
    </div>
    """, unsafe_allow_html=True)

    auto_email = config.get("auto_email", False)
    if auto_email:
        st.markdown(
            "<div style='color: #34d399; font-size: 0.82rem; margin-bottom: 0.5rem;'>"
            "📧 Auto-send filter rejections is <strong>ON</strong> (toggle in sidebar)</div>",
            unsafe_allow_html=True,
        )

    analyze_btn = st.button(
        "🚀 Analyze & Rank Candidates",
        type="primary",
        use_container_width=True,
        key="analyze_btn",
    )

    # ── Validation & Pipeline Trigger ──────────────────────────────────────────
    if analyze_btn:
        if not job_description.strip():
            st.error("⚠️ Please paste a Job Description before running the pipeline.")
            st.stop()

        if config["data_source"] == "📂 Folder Scan":
            _scan_folder = Path(config["folder_path"]) if config["folder_path"] else Path(__file__).parent / "test_cvs"
            if not _scan_folder.is_dir():
                st.error(f"⚠️ Folder not found: `{_scan_folder}`. Update the path in the sidebar.")
                st.stop()

        if config["data_source"] == "📁 Upload PDFs" and not uploaded_files:
            st.error("⚠️ Please upload at least one PDF resume.")
            st.stop()

        if config["data_source"] == "🔗 Google Sheet" and not config["sheet_id"]:
            st.error("⚠️ Please enter a Google Sheet ID in the sidebar.")
            st.stop()

        # Load the embedding model (cached — fast after first load)
        with st.spinner("🔄 Loading embedding model..."):
            model = _get_model(config["model_name"])

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)
        st.markdown("### 📊 Pipeline Running...")

        # ── Execute the appropriate pipeline ──────────────────────────────────
        if config["data_source"] == "📂 Folder Scan":
            _scan_folder = Path(config["folder_path"]) if config["folder_path"] else Path(__file__).parent / "test_cvs"
            results_df, stats, candidates, filtered = _run_pipeline_folder_mode(
                jd=job_description,
                folder_path=str(_scan_folder),
                model=model,
            )
        elif config["data_source"] == "📁 Upload PDFs":
            results_df, stats, candidates, filtered = _run_pipeline_upload_mode(
                jd=job_description,
                uploaded_files=uploaded_files,
                model=model,
            )
        else:
            results_df, stats, candidates, filtered = _run_pipeline_sheet_mode(
                jd=job_description,
                sheet_id=config["sheet_id"],
                credentials_path=config["credentials_path"],
                min_cgpa=config["min_cgpa"],
                min_years_exp=config["min_years_exp"],
                model=model,
            )

        # Persist results in session state so they survive Streamlit reruns
        st.session_state["results_df"] = results_df
        st.session_state["stats"]      = stats
        st.session_state["candidates"] = candidates
        st.session_state["filtered"]   = filtered
        st.session_state["job_description"] = job_description

        # Save to disk cache for persistence across full browser reloads
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({
                    "results_df": results_df,
                    "stats": stats,
                    "candidates": candidates,
                    "filtered": filtered,
                    "job_description": job_description,
                }, f)
        except Exception as e:
            pass

        # Auto-send filter rejection emails if toggled
        if auto_email and filtered:
            with st.spinner("📧 Auto-sending filter rejection emails..."):
                send_filter_rejection_emails(
                    filtered_candidates=filtered,
                    position=config.get("position_name", "the open position"),
                    company_name=config.get("company_name", "Our Organization"),
                    dry_run=False,
                )
            st.toast("✅ Auto-sent filter rejection emails!")

    # ── Results Section (shown if analysis has been run) ──────────────────────
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        stats      = st.session_state["stats"]
        candidates = st.session_state["candidates"]
        filtered   = st.session_state["filtered"]
        jd_text    = st.session_state.get("job_description", "")

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)
        st.success(
            f"✅ Analysis complete — **{stats['n_candidates']}** candidate(s) ranked  "
            f"| {len(filtered)} filtered out  "
            f"| Top score: **{stats['top_score']:.1f}%**"
        )

        # ── KPI Metrics ───────────────────────────────────────────────────────
        _render_kpi_row(stats, n_filtered=len(filtered), n_total=stats["n_candidates"] + len(filtered))

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Rankings Table + Chart ─────────────────────────────────────────────
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown("<div class='section-heading'>🏆 Ranked Results</div>", unsafe_allow_html=True)
            _render_ranked_table(results_df)

        with right_col:
            st.markdown("<div class='section-heading'>📊 Score Distribution</div>", unsafe_allow_html=True)
            _render_score_chart(results_df)

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Candidate Detail Cards ────────────────────────────────────────────
        _render_candidate_details(results_df, candidates)

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Filtered Candidates Panel ─────────────────────────────────────────
        _render_filtered_candidates(filtered)

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── AI HR Chatbot ─────────────────────────────────────────────────────
        _render_chatbot(
            results_df=results_df,
            candidates=candidates,
            job_description=jd_text,
            filtered=filtered,
        )

        st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

        # ── Candidate Selection & Email Dispatch ──────────────────────────────
        _render_candidate_selection(
            results_df=results_df,
            candidates=candidates,
            filtered=filtered,
        )

    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)

    # ── Pipeline Scheduling ───────────────────────────────────────────────────
    _render_scheduling()

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("<hr class='glass-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#334155; font-size:0.72rem; padding-bottom:1rem;'>
        CVision Phase 2 · Built with Streamlit · BERT + Gemini AI · pdfplumber · Google Forms API
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
