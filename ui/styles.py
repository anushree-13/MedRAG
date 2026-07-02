"""
ui/styles.py — Glassmorphism CSS Injection

Injects custom CSS into Streamlit for:
  - Animated gradient background
  - Glass-panel cards with backdrop-filter blur
  - Custom sidebar styling
  - Button hover effects, status indicators, scrollbar
  - Google Fonts (Inter)
"""

import streamlit as st


def inject_styles():
    """Inject the full glassmorphism CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ───────────────────────────────────────────── */
:root {
    --bg-primary: #0F0F1A;
    --bg-secondary: #1A1A2E;
    --bg-glass: rgba(26, 26, 46, 0.6);
    --border-glass: rgba(124, 58, 237, 0.2);
    --accent-purple: #7C3AED;
    --accent-cyan: #06B6D4;
    --accent-green: #10B981;
    --accent-amber: #F59E0B;
    --accent-pink: #EC4899;
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
}

/* ── Global ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Glass Panel ──────────────────────────────────────────────── */
.glass-panel {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glass-panel:hover {
    border-color: rgba(124, 58, 237, 0.4);
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.15);
    transform: translateY(-2px);
}

/* ── Glass Panel Variants ─────────────────────────────────────── */
.glass-panel-cyan {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.glass-panel-green {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.15);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Status Indicators ────────────────────────────────────────── */
.status-idle    { color: #6B7280; }
.status-running { color: #3B82F6; animation: pulse 1.5s infinite; }
.status-done    { color: #10B981; }
.status-error   { color: #EF4444; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
}

.status-badge-idle    { background: rgba(107,114,128,0.15); color: #9CA3AF; }
.status-badge-running { background: rgba(59,130,246,0.15);  color: #60A5FA; }
.status-badge-done    { background: rgba(16,185,129,0.15);  color: #34D399; }
.status-badge-error   { background: rgba(239,68,68,0.15);   color: #F87171; }

/* ── Confidence Badge ─────────────────────────────────────────── */
.confidence-high   { background: rgba(16,185,129,0.2); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.confidence-medium { background: rgba(245,158,11,0.2);  color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
.confidence-low    { background: rgba(239,68,68,0.2);   color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

.confidence-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    color: var(--text-secondary) !important;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    border-color: var(--accent-purple) !important;
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.1)) !important;
    border-color: var(--accent-purple) !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), #5B21B6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
}

/* ── Chat Messages ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    margin-bottom: 0.5rem;
}

/* ── Expanders ────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: rgba(124, 58, 237, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(124, 58, 237, 0.5);
}

/* ── Metric Cards ─────────────────────────────────────────────── */
.metric-card {
    background: var(--bg-glass);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-card .metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(124, 58, 237, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(124, 58, 237, 0.05) !important;
}

/* ── Progress bar ─────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan)) !important;
    border-radius: 10px;
}
</style>
"""
