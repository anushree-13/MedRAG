"""
ui/chat_view.py — Chat Answer Panel

Features:
  - Chat interface with st.chat_message
  - Streamed answer display
  - Expandable reasoning trace
  - Source citations with chunk previews
  - Validation badge (faithfulness + confidence)
  - Session-based conversation history
"""

import streamlit as st
from core.models import AgentState, GeneratedResponse, ValidationReport, SearchResult


def render_chat_view(orchestrator):
    """Render the chat interface panel."""

    st.markdown(
        '<h2 style="background: linear-gradient(135deg, #7C3AED, #06B6D4); '
        '-webkit-background-clip: text; -webkit-text-fill-color: transparent;">'
        '💬 Research Chat</h2>',
        unsafe_allow_html=True,
    )

    # ── Initialize conversation history ───────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Display conversation history ──────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "extra" in msg:
                _render_extras(msg["extra"])

    # ── Chat input ────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask a question about your research papers...",
        key="chat_input",
    )

    if user_input:
        # Add user message
        st.session_state["messages"].append({
            "role": "user", "content": user_input,
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            _handle_query(user_input, orchestrator)


def _handle_query(query: str, orchestrator):
    """Process a query through the orchestrator and display results."""

    # Check if indexes are loaded
    if not orchestrator.indexing_agent or not orchestrator.indexing_agent.is_loaded:
        st.warning("⚠️ Please upload and process documents first.")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": "⚠️ Please upload and process documents first.",
        })
        return

    with st.spinner("🔍 Searching and analyzing..."):
        try:
            response, validation, results = orchestrator.query(query)
        except Exception as e:
            error_msg = str(e)
            status_code = getattr(e, "status_code", None)

            # ── 413: Request too large ────────────────────────────────
            if status_code == 413 or "413" in error_msg:
                st.error(
                    "📦 **Request Too Large** — The retrieved context exceeded "
                    "the model's input limit.\n\n"
                    "**Try one of these:**\n"
                    "- Ask a more specific or narrower question\n"
                    "- Upload shorter / less dense documents\n"
                )
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "⚠️ The query produced too much context for the model. "
                               "Please try a more specific question.",
                })
                return

            # ── 401: Authentication error ─────────────────────────────
            if status_code == 401 or "401" in error_msg:
                st.error(
                    "🔑 **Authentication Failed** — Your Groq API key appears "
                    "invalid or expired.\n\n"
                    "Check `GROQ_API_KEY` in your `.env` file and restart the app."
                )
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "⚠️ API authentication failed. Please verify your "
                               "GROQ_API_KEY in the .env file.",
                })
                return

            # ── Generic fallback ──────────────────────────────────────
            st.error(f"❌ Error: {e}")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"❌ Error processing query: {e}",
            })
            return

    # ── Display answer ────────────────────────────────────────────────
    st.markdown(response.answer)

    extra = {
        "response": response,
        "validation": validation,
        "results": results,
    }
    _render_extras(extra)

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": response.answer,
        "extra": extra,
    })


def _render_extras(extra: dict):
    """Render validation badge, reasoning trace, and source citations."""

    response: GeneratedResponse = extra.get("response")
    validation: ValidationReport = extra.get("validation")
    results: list[SearchResult] = extra.get("results", [])

    if not response:
        return

    # ── Validation Badge ──────────────────────────────────────────────
    if validation:
        faith = validation.faithfulness_score
        conf = validation.final_confidence

        if conf >= 0.7:
            badge_class = "confidence-high"
            label = "High Confidence"
        elif conf >= 0.4:
            badge_class = "confidence-medium"
            label = "Medium Confidence"
        else:
            badge_class = "confidence-low"
            label = "Low Confidence"

        st.markdown(
            f'<div style="margin: 0.8rem 0;">'
            f'<span class="confidence-badge {badge_class}">'
            f'🎯 {label} ({conf:.0%})</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color: #94A3B8; font-size: 0.85rem;">'
            f'Faithfulness: {faith:.0%} '
            f'({validation.faithful_claims}/{validation.total_claims} claims verified)'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # ── Reasoning Trace ───────────────────────────────────────────────
    if response.reasoning_steps:
        with st.expander("🧠 Reasoning Trace", expanded=False):
            for step in response.reasoning_steps:
                st.markdown(
                    f'<div class="glass-panel" style="padding:0.6rem; margin-bottom:0.4rem;">'
                    f'<span style="color:#7C3AED; font-weight:600;">Step {step.step_num}:</span> '
                    f'<span style="color:#06B6D4;">{step.action}</span><br>'
                    f'<span style="color:#94A3B8; font-size:0.85rem;">{step.detail}</span>'
                    f'{"<br><span style=color:#64748B;font-size:0.75rem;>⏱ " + f"{step.duration_ms:.0f}ms</span>" if step.duration_ms else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Reflection Notes ──────────────────────────────────────────────
    if validation and validation.reflection_notes:
        with st.expander("🔍 Self-Reflection", expanded=False):
            for note in validation.reflection_notes:
                st.markdown(f"- {note}")

    # ── Source Citations ──────────────────────────────────────────────
    if results:
        with st.expander(f"📚 Sources ({len(results)} chunks)", expanded=False):
            for i, r in enumerate(results[:5]):
                c = r.chunk
                source_color = {
                    "hybrid": "#7C3AED",
                    "semantic": "#06B6D4",
                    "keyword": "#10B981",
                }.get(r.source, "#6B7280")

                st.markdown(
                    f'<div class="glass-panel" style="padding:0.8rem;">'
                    f'<div style="display:flex; justify-content:space-between; '
                    f'margin-bottom:0.4rem;">'
                    f'<span style="color:#E2E8F0; font-weight:500;">'
                    f'📄 {c.doc_filename} — p.{c.page_num + 1}</span>'
                    f'<span style="color:{source_color}; font-size:0.8rem;">'
                    f'⬤ {r.source} (rank #{r.rank})</span>'
                    f'</div>'
                    f'<p style="color:#94A3B8; font-size:0.85rem; margin:0;">'
                    f'{c.text[:300]}{"..." if len(c.text) > 300 else ""}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
