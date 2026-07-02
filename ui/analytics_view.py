"""
ui/analytics_view.py — Analytics Dashboard Panel

Features:
  - Comparison tables (methods × metrics)
  - Dataset & method frequency charts
  - Gap analysis: UMAP cluster scatter + gap descriptions
  - Cluster size vs density chart
  - CSV/PNG export buttons
"""

import streamlit as st
import numpy as np

from agents.synthesis.visualization import VisualizationAgent
from core.models import GapReport


def render_analytics_view(orchestrator):
    """Render the analytics dashboard panel."""

    st.markdown(
        '<h2 style="background: linear-gradient(135deg, #F59E0B, #EC4899); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        '📊 Analytics Dashboard</h2>',
        unsafe_allow_html=True,
    )

    viz = VisualizationAgent()
    entities = orchestrator.all_entities

    # ── Overview Metrics ──────────────────────────────────────────────
    datasets = [e for e in entities if e.entity_type == "dataset"]
    metrics = [e for e in entities if e.entity_type == "metric"]
    methods = [e for e in entities if e.entity_type == "method"]
    hardware = [e for e in entities if e.entity_type == "hardware"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(set(d.name for d in datasets))}</div>'
            f'<div class="metric-label">Datasets</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(metrics)}</div>'
            f'<div class="metric-label">Metrics</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(set(m.name for m in methods))}</div>'
            f'<div class="metric-label">Methods</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(hardware)}</div>'
            f'<div class="metric-label">HW Mentions</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if not entities:
        st.markdown(
            '<div class="glass-panel" style="text-align:center; padding:3rem;">'
            '<p style="font-size:3rem; margin-bottom:0.5rem;">📊</p>'
            '<p style="color:#94A3B8;">No analytics data yet.</p>'
            '<p style="color:#64748B; font-size:0.85rem;">'
            'Ask questions in the Chat tab to trigger entity extraction.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Charts ────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 Comparisons", "📈 Distributions", "🗺️ Gap Analysis"])

    with tab1:
        st.markdown("#### Metrics Comparison Table")
        fig = viz.comparison_table(entities)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Dataset Usage")
            fig_ds = viz.dataset_distribution(entities)
            st.plotly_chart(fig_ds, use_container_width=True)

        with col_b:
            st.markdown("#### Method Frequency")
            fig_mf = viz.method_frequency(entities)
            st.plotly_chart(fig_mf, use_container_width=True)

    with tab3:
        _render_gap_analysis(orchestrator, viz)


def _render_gap_analysis(orchestrator, viz: VisualizationAgent):
    """Render gap analysis section with clustering scatter + descriptions."""

    indexer = orchestrator.indexing_agent
    if indexer is None or not indexer.is_loaded:
        st.info("Upload and process documents to enable gap analysis.")
        return

    # ── Run or load gap analysis ──────────────────────────────────────
    if "gap_report" not in st.session_state:
        if st.button("🔬 Run Gap Analysis", use_container_width=True):
            with st.spinner("Clustering embeddings..."):
                try:
                    from agents.analysis.gap_detection import GapDetectionAgent
                    agent = GapDetectionAgent()
                    embeddings = indexer.get_all_embeddings()
                    chunk_ids = indexer.chunk_ids
                    texts = [indexer.chunk_map[c].text for c in chunk_ids]
                    report = agent.detect(embeddings, chunk_ids, texts)
                    st.session_state["gap_report"] = report
                    st.session_state["gap_embeddings"] = embeddings
                    st.session_state["gap_texts"] = texts
                    st.rerun()
                except Exception as e:
                    st.error(f"Gap analysis failed: {e}")
        return

    report: GapReport = st.session_state["gap_report"]
    embeddings = st.session_state.get("gap_embeddings")
    texts = st.session_state.get("gap_texts")

    # ── Cluster stats ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="glass-panel-cyan" style="padding:1rem;">'
        f'<span style="color:#06B6D4; font-weight:600;">Optimal Clusters:</span> '
        f'{report.optimal_k} &nbsp;|&nbsp; '
        f'<span style="color:#06B6D4; font-weight:600;">Silhouette Score:</span> '
        f'{report.silhouette_score:.3f}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Cluster scatter ───────────────────────────────────────────────
    if embeddings is not None and report.clusters:
        # Reconstruct labels from cluster info
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=report.optimal_k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)

        fig = viz.cluster_scatter(embeddings, labels, report.clusters, texts)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster size vs density
        fig2 = viz.gap_summary_chart(report)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Gap Descriptions ──────────────────────────────────────────────
    st.markdown("#### 🔍 Identified Research Gaps")
    for i, gap in enumerate(report.gaps):
        icon = "🟡" if "under-explored" in gap.lower() else (
            "🟠" if "niche" in gap.lower() else "🔵"
        )
        st.markdown(
            f'<div class="glass-panel" style="padding:1rem;">'
            f'<span style="font-size:1.1rem;">{icon}</span> '
            f'<span style="color:#E2E8F0;">{gap}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Cluster Themes ────────────────────────────────────────────────
    if report.clusters:
        st.markdown("#### 🏷️ Cluster Themes")
        for c in report.clusters:
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; margin:4px 0;">'
                f'<span style="color:#7C3AED; font-weight:600;">C{c.cluster_id}:</span>'
                f'<span style="color:#CBD5E1;">{c.theme}</span>'
                f'<span style="color:#64748B; font-size:0.8rem;">({c.size} chunks, '
                f'density={c.density:.3f})</span></div>',
                unsafe_allow_html=True,
            )
