"""
app.py — Agentic Hybrid RAG Dashboard

Main Streamlit entry point. Orchestrates:
  - Glassmorphism UI with animated background
  - Dynamic sidebar for uploads and agent monitoring
  - Multi-view tabs: Chat | Knowledge Graph | Analytics
  - Real-time agent status updates during processing
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# ── Ensure project root is on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.orchestrator import Orchestrator
from core.models import AgentStatus
from ui.styles import inject_styles
from ui.sidebar import render_sidebar, update_agent_status
from ui.chat_view import render_chat_view
from ui.graph_view import render_graph_view
from ui.analytics_view import render_analytics_view

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Hybrid RAG",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Agentic Hybrid RAG — Multi-Agent Research Intelligence System",
    },
)

# ── Inject Styles ─────────────────────────────────────────────────────
inject_styles()


# ── Initialize Orchestrator ───────────────────────────────────────────

def _get_orchestrator() -> Orchestrator:
    """Get or create the session orchestrator."""
    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = Orchestrator(
            on_status=update_agent_status,
        )
    return st.session_state["orchestrator"]


orchestrator = _get_orchestrator()


# ── Sidebar ───────────────────────────────────────────────────────────
render_sidebar()


# ── Handle Ingestion Trigger ──────────────────────────────────────────
if st.session_state.get("trigger_ingest"):
    st.session_state["trigger_ingest"] = False

    uploaded_files = st.session_state.get("uploaded_files", [])
    if uploaded_files:
        settings = get_settings()
        saved_paths: list[Path] = []

        # Save uploaded files to disk
        for uf in uploaded_files:
            dest = settings.upload_dir / uf.name
            dest.write_bytes(uf.getbuffer())
            saved_paths.append(dest)

        # Run ingestion pipeline
        with st.spinner("🔄 Processing documents..."):
            try:
                chunks = orchestrator.ingest(saved_paths)
                st.session_state["docs_count"] = st.session_state.get("docs_count", 0) + len(uploaded_files)
                st.session_state["chunks_count"] = st.session_state.get("chunks_count", 0) + len(chunks)
                st.toast(f"✅ Processed {len(uploaded_files)} document(s) → {len(chunks)} chunks", icon="🎉")
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")
                logger.error(f"Ingestion error: {e}", exc_info=True)

        st.rerun()


# ── Main Content Area ─────────────────────────────────────────────────

# Header
st.markdown(
    '<div style="text-align:center; margin-bottom:1.5rem;">'
    '<h1 style="background: linear-gradient(135deg, #7C3AED, #06B6D4, #10B981); '
    '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
    'font-size: 2.2rem; margin-bottom:0;">🧬 Agentic Hybrid RAG</h1>'
    '<p style="color: #64748B; font-size:0.95rem; margin-top:0.3rem;">'
    'Multi-Agent Research Intelligence — Powered by FAISS + BM25 + LLaMA 3'
    '</p></div>',
    unsafe_allow_html=True,
)

# Tab navigation
tab_chat, tab_graph, tab_analytics = st.tabs([
    "💬 Chat",
    "🕸️ Knowledge Graph",
    "📊 Analytics",
])

with tab_chat:
    render_chat_view(orchestrator)

with tab_graph:
    render_graph_view(orchestrator)

with tab_analytics:
    render_analytics_view(orchestrator)


# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; margin-top:3rem; padding:1rem; '
    'border-top:1px solid rgba(124,58,237,0.15);">'
    '<p style="color:#4B5563; font-size:0.75rem;">'
    '🧬 Agentic Hybrid RAG • FAISS + BM25 + LLaMA 3 via Groq • '
    'Built with Streamlit</p></div>',
    unsafe_allow_html=True,
)
