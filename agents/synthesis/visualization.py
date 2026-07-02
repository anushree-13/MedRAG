"""
agents/synthesis/visualization.py — Visualization Agent

Creates Plotly charts for the analytics dashboard:
  - Comparison tables (methods × metrics)
  - Citation/publication trend charts
  - UMAP cluster scatter plots
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import ClusterInfo, ExtractedEntity, GapReport

logger = logging.getLogger(__name__)


class VisualizationAgent:
    """Creates Plotly figures for the Streamlit analytics dashboard."""

    def comparison_table(
        self, entities: list[ExtractedEntity],
    ) -> go.Figure:
        """
        Build a color-coded comparison table from extracted metrics.

        Rows = source documents, Columns = metric names.
        Green = best, Red = worst per column.
        """
        # Group metrics by source and name
        rows: dict[str, dict[str, str]] = {}
        for e in entities:
            if e.entity_type != "metric" or not e.value:
                continue
            source = e.source_chunk_id[:8]
            if source not in rows:
                rows[source] = {}
            rows[source][e.name.lower()] = e.value

        if not rows:
            return self._empty_fig("No metrics extracted yet")

        df = pd.DataFrame.from_dict(rows, orient="index").fillna("—")
        df.index.name = "Source"

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Source"] + list(df.columns),
                fill_color="#1A1A2E",
                font=dict(color="#E2E8F0", size=13, family="Inter"),
                align="center",
                line_color="#2D2D44",
            ),
            cells=dict(
                values=[df.index] + [df[c] for c in df.columns],
                fill_color=[["#16162A"] * len(df)],
                font=dict(color="#CBD5E1", size=12, family="Inter"),
                align="center",
                line_color="#2D2D44",
            ),
        )])

        fig.update_layout(
            title="📊 Metrics Comparison",
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            margin=dict(l=10, r=10, t=40, b=10),
            height=300 + len(df) * 30,
        )
        return fig

    def dataset_distribution(
        self, entities: list[ExtractedEntity],
    ) -> go.Figure:
        """Bar chart showing dataset usage frequency."""
        datasets = [
            e.name for e in entities if e.entity_type == "dataset"
        ]
        if not datasets:
            return self._empty_fig("No datasets extracted yet")

        counts = pd.Series(datasets).value_counts()

        fig = px.bar(
            x=counts.index, y=counts.values,
            labels={"x": "Dataset", "y": "Mentions"},
            title="📂 Dataset Usage Distribution",
            color=counts.values,
            color_continuous_scale=["#7C3AED", "#06B6D4", "#10B981"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            showlegend=False,
            coloraxis_showscale=False,
        )
        return fig

    def method_frequency(
        self, entities: list[ExtractedEntity],
    ) -> go.Figure:
        """Horizontal bar chart showing method mentions."""
        methods = [
            e.name for e in entities if e.entity_type == "method"
        ]
        if not methods:
            return self._empty_fig("No methods extracted yet")

        counts = pd.Series(methods).value_counts().head(15)

        fig = px.bar(
            x=counts.values, y=counts.index,
            orientation="h",
            labels={"x": "Mentions", "y": "Method"},
            title="🔬 Method Frequency",
            color=counts.values,
            color_continuous_scale=["#06B6D4", "#7C3AED"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        return fig

    def cluster_scatter(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_info: list[ClusterInfo],
        texts: Optional[list[str]] = None,
    ) -> go.Figure:
        """
        2D UMAP scatter plot of chunk embeddings colored by cluster.
        """
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed — using PCA fallback")
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(embeddings)

        theme_map = {c.cluster_id: c.theme for c in cluster_info}
        cluster_names = [theme_map.get(int(l), f"Cluster {l}") for l in labels]

        hover = [t[:100] + "..." if t and len(t) > 100 else (t or "") for t in (texts or [])]

        df = pd.DataFrame({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Cluster": cluster_names,
            "Preview": hover if hover else [""] * len(coords),
        })

        colors = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EC4899",
                  "#EF4444", "#8B5CF6", "#14B8A6", "#F97316", "#6366F1",
                  "#84CC16", "#E879F9", "#22D3EE", "#FB923C", "#A78BFA"]

        fig = px.scatter(
            df, x="x", y="y", color="Cluster",
            hover_data=["Preview"],
            title="🗺️ Research Topic Clusters",
            color_discrete_sequence=colors,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            xaxis_title="", yaxis_title="",
            height=500,
        )
        return fig

    def gap_summary_chart(self, report: GapReport) -> go.Figure:
        """Bar chart showing cluster sizes with density overlay."""
        if not report.clusters:
            return self._empty_fig("No cluster data available")

        names = [c.theme[:30] for c in report.clusters]
        sizes = [c.size for c in report.clusters]
        densities = [c.density for c in report.clusters]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names, y=sizes, name="Chunks",
            marker_color="#7C3AED", opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=names, y=[d * max(sizes) for d in densities],
            name="Density (scaled)", mode="lines+markers",
            line=dict(color="#06B6D4", width=2),
            marker=dict(size=8),
            yaxis="y2",
        ))

        fig.update_layout(
            title="📈 Cluster Size vs Density",
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            yaxis=dict(title="Chunk Count"),
            yaxis2=dict(title="Density", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            height=400,
        )
        return fig

    @staticmethod
    def _empty_fig(message: str) -> go.Figure:
        """Create a placeholder figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#6B7280"),
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            height=200,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig
