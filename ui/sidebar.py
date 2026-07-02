"""
ui/sidebar.py — Dynamic Sidebar Component

Features:
  - PDF file upload (drag-and-drop, multi-file)
  - "Process Documents" trigger button
  - Live agent status monitor with animated indicators
  - Document list with stats
  - Tunable settings panel
"""

import streamlit as st
from pathlib import Path
from core.config import get_settings
from core.models import AgentState, AgentStatus


_STATUS_ICONS = {
    AgentState.IDLE:    "⚪",
    AgentState.RUNNING: "🔵",
    AgentState.DONE:    "🟢",
    AgentState.ERROR:   "🔴",
}

_STATUS_CSS = {
    AgentState.IDLE:    "status-badge-idle",
    AgentState.RUNNING: "status-badge-running",
    AgentState.DONE:    "status-badge-done",
    AgentState.ERROR:   "status-badge-error",
}

# Agent names for the status monitor
_AGENTS = [
    "Parsing Agent", "Chunking Agent", "Indexing Agent",
    "Query Classifier", "Retrieval Agent", "Extraction Agent",
    "LLM Generator", "Validation Agent",
    "Gap Detection", "Knowledge Graph",
]


def render_sidebar():
    """Render the full sidebar with upload, status, and settings."""
    with st.sidebar:
        # ── Logo / Title ──────────────────────────────────────────────
        st.markdown(
            '<h1 style="text-align:center; background: linear-gradient(135deg, '
            '#7C3AED, #06B6D4); -webkit-background-clip: text; '
            '-webkit-text-fill-color: transparent; font-size: 1.6rem; '
            'margin-bottom: 0;">🧬 Agentic RAG</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center; color: #64748B; font-size:0.8rem; '
            'margin-top:0;">Hybrid Research Intelligence</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        # ── File Upload ───────────────────────────────────────────────
        st.markdown("### 📄 Upload Papers")
        uploaded_files = st.file_uploader(
            "Drop clinical/research PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(
                f'<div class="glass-panel" style="padding:0.8rem;">'
                f'<span style="color:#10B981;">✓</span> '
                f'{len(uploaded_files)} file(s) selected</div>',
                unsafe_allow_html=True,
            )

        # ── Process Button ────────────────────────────────────────────
        process_clicked = st.button(
            "⚡ Process Documents",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if process_clicked and uploaded_files:
            st.session_state["trigger_ingest"] = True
            st.session_state["uploaded_files"] = uploaded_files

        st.divider()

        # ── Agent Status Monitor ──────────────────────────────────────
        st.markdown("### 🤖 Agent Status")

        if "agent_statuses" not in st.session_state:
            st.session_state["agent_statuses"] = {
                name: AgentStatus(name=name) for name in _AGENTS
            }

        for name in _AGENTS:
            status = st.session_state["agent_statuses"].get(
                name, AgentStatus(name=name)
            )
            icon = _STATUS_ICONS.get(status.state, "⚪")
            css_class = _STATUS_CSS.get(status.state, "status-badge-idle")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#CBD5E1;">'
                    f'{icon} {name}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="status-badge {css_class}" '
                    f'style="font-size:0.7rem;">{status.state.value}</div>',
                    unsafe_allow_html=True,
                )

            if status.state == AgentState.RUNNING and status.progress > 0:
                st.progress(status.progress)

            if status.message and status.state in (AgentState.DONE, AgentState.ERROR):
                st.caption(status.message)

        st.divider()

        # ── Document Stats ────────────────────────────────────────────
        st.markdown("### 📊 Corpus Stats")
        docs_count = st.session_state.get("docs_count", 0)
        chunks_count = st.session_state.get("chunks_count", 0)
        entities_count = st.session_state.get("entities_count", 0)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{docs_count}</div>'
                f'<div class="metric-label">Docs</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{chunks_count}</div>'
                f'<div class="metric-label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{entities_count}</div>'
                f'<div class="metric-label">Entities</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Settings ──────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            settings = get_settings()
            st.session_state["top_k"] = st.slider(
                "Top-K Results", 3, 20, settings.top_k,
            )
            st.session_state["rrf_k"] = st.slider(
                "RRF Smoothing (k)", 10, 100, settings.rrf_k,
            )
            st.markdown(
                f"**Model:** `{settings.llm_model}`  \n"
                f"**Embeddings:** `{settings.embedding_model}`  \n"
                f"**Chunk:** {settings.chunk_size}/{settings.chunk_overlap}",
            )


def update_agent_status(status: AgentStatus):
    """Callback to update agent status from the orchestrator."""
    if "agent_statuses" in st.session_state:
        st.session_state["agent_statuses"][status.name] = status
