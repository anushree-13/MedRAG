"""
ui/graph_view.py — Knowledge Graph Panel

Features:
  - Interactive knowledge graph via streamlit-agraph
  - Color-coded nodes by type (Paper, Method, Dataset, Metric, Result)
  - Graph statistics panel
  - Node type filter controls
"""

import streamlit as st

from agents.synthesis.knowledge_graph import KnowledgeGraphAgent, NODE_COLORS


def render_graph_view(orchestrator):
    """Render the knowledge graph visualization panel."""

    st.markdown(
        '<h2 style="background: linear-gradient(135deg, #06B6D4, #10B981); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        '🕸️ Knowledge Graph</h2>',
        unsafe_allow_html=True,
    )

    # Check if a graph exists
    kg_agent = orchestrator.kg_agent
    if kg_agent is None:
        kg_agent = KnowledgeGraphAgent()

    graph_path = kg_agent._graph_dir / "knowledge_graph.graphml"
    if not graph_path.exists():
        st.markdown(
            '<div class="glass-panel" style="text-align:center; padding:3rem;">'
            '<p style="font-size:3rem; margin-bottom:0.5rem;">🕸️</p>'
            '<p style="color:#94A3B8;">No knowledge graph yet.</p>'
            '<p style="color:#64748B; font-size:0.85rem;">'
            'Upload papers and run analysis to build the graph.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🔬 Run Analysis", use_container_width=True):
            with st.spinner("Building knowledge graph..."):
                try:
                    orchestrator.analyze()
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        return

    # ── Load the graph ────────────────────────────────────────────────
    try:
        G = kg_agent.load_graph()
    except Exception as e:
        st.error(f"Failed to load graph: {e}")
        return

    stats = kg_agent.get_stats(G)

    # ── Stats Row ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{stats["total_nodes"]}</div>'
            f'<div class="metric-label">Nodes</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{stats["total_edges"]}</div>'
            f'<div class="metric-label">Edges</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(stats["node_types"])}</div>'
            f'<div class="metric-label">Types</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{len(stats["relationship_types"])}</div>'
            f'<div class="metric-label">Relations</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Filter Controls ───────────────────────────────────────────────
    with st.expander("🎛️ Filters", expanded=False):
        available_types = list(stats["node_types"].keys())
        selected_types = st.multiselect(
            "Show node types:",
            available_types,
            default=available_types,
            key="graph_filter_types",
        )

    # ── Render Graph ──────────────────────────────────────────────────
    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        # Filter graph nodes by type
        nodes = []
        visible_ids = set()
        for nid, data in G.nodes(data=True):
            ntype = data.get("node_type", "Unknown")
            if ntype not in selected_types:
                continue
            visible_ids.add(nid)
            nodes.append(Node(
                id=nid,
                label=nid if len(nid) <= 30 else nid[:27] + "...",
                size=20 + G.degree(nid) * 3,
                color=NODE_COLORS.get(ntype, NODE_COLORS["Unknown"]),
                title=f"Type: {ntype}\nConnections: {G.degree(nid)}",
            ))

        edges = []
        for u, v, data in G.edges(data=True):
            if u in visible_ids and v in visible_ids:
                edges.append(Edge(
                    source=u, target=v,
                    label=data.get("label", ""),
                    color="#4B556380",
                ))

        config = Config(
            width=800,
            height=500,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#7C3AED",
            collapsible=True,
            node={"labelProperty": "label"},
            link={"labelProperty": "label", "renderLabel": True},
        )

        agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        st.warning(
            "⚠️ `streamlit-agraph` not installed. "
            "Run `pip install streamlit-agraph` to enable graph visualization."
        )
        _render_text_fallback(G, stats)

    # ── Legend ────────────────────────────────────────────────────────
    st.markdown("")
    legend_html = '<div style="display:flex; gap:1rem; flex-wrap:wrap; justify-content:center;">'
    for ntype, color in NODE_COLORS.items():
        if ntype == "Unknown":
            continue
        legend_html += (
            f'<span style="display:inline-flex; align-items:center; gap:4px;">'
            f'<span style="width:12px; height:12px; border-radius:50%; '
            f'background:{color}; display:inline-block;"></span>'
            f'<span style="color:#94A3B8; font-size:0.8rem;">{ntype}</span></span>'
        )
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    # ── Most Connected Entities ───────────────────────────────────────
    if stats["most_connected"]:
        st.markdown("")
        st.markdown("#### 🔗 Most Connected Entities")
        for item in stats["most_connected"][:5]:
            ntype = G.nodes[item["node"]].get("node_type", "?") if item["node"] in G.nodes else "?"
            color = NODE_COLORS.get(ntype, "#6B7280")
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; margin:4px 0;">'
                f'<span style="color:{color};">●</span>'
                f'<span style="color:#E2E8F0;">{item["node"]}</span>'
                f'<span style="color:#64748B; font-size:0.8rem;">({item["degree"]} connections)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_text_fallback(G, stats: dict):
    """Simple text-based fallback when agraph is unavailable."""
    st.markdown("**Node Types:**")
    for ntype, count in stats["node_types"].items():
        color = NODE_COLORS.get(ntype, "#6B7280")
        st.markdown(
            f'<span style="color:{color};">●</span> {ntype}: {count}',
            unsafe_allow_html=True,
        )

    st.markdown("**Top Relationships:**")
    for rel, count in sorted(
        stats["relationship_types"].items(), key=lambda x: -x[1]
    )[:10]:
        st.markdown(f"- {rel}: {count}")
