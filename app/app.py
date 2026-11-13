import sys
import uuid
from html import escape
from pathlib import Path

import streamlit as st

# 1. PROJECT CONFIGURATION

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_DIR / "app"

# Add both directories to Python's import path
for directory in [PROJECT_DIR, APP_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Import ConductIQ agent
from agent import ask_conductiq_detailed

# Import feedback manager
from core.feedback_manager import save_feedback


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="ConductIQ | RUB Student Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# 3. CUSTOM STYLING

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --brand: #2563EB;
    --brand-dark: #1D4ED8;
    --brand-soft: rgba(37, 99, 235, 0.08);
    --accent: #7C3AED;
    --muted: #64748b;
    --border: rgba(100, 116, 139, 0.25);
    --card-bg: rgba(148, 163, 184, 0.08);
    --ok: #16a34a;
    --warn: #d97706;
}

html, body, [class*="css"], .stMarkdown, .stChatInput textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Page width and spacing */
.block-container {
    max-width: 780px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* ── Hero header ───────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #7C3AED 130%);
    border-radius: 20px;
    padding: 28px 28px 24px 28px;
    margin-bottom: 14px;
    color: #ffffff;
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -40px; top: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.10);
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.18);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    color: #ffffff;
}
.hero-subtitle {
    font-size: 1rem;
    margin-top: 6px;
    color: rgba(255, 255, 255, 0.85);
}

/* ── Disclaimer ────────────────────────────────────────── */
.disclaimer {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--brand);
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 0.82rem;
    line-height: 1.5;
    color: var(--muted);
    margin-bottom: 22px;
}
.disclaimer strong { color: inherit; }

/* ── Welcome section ───────────────────────────────────── */
.welcome-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 6px 0 2px 0;
}
.welcome-text {
    color: var(--muted);
    margin-bottom: 16px;
}
.example-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.example-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 0.92rem;
    transition: all 0.15s ease;
}
.example-card:hover {
    border-color: var(--brand);
    background: var(--brand-soft);
    transform: translateY(-1px);
}
.example-icon {
    font-size: 1.05rem;
    flex-shrink: 0;
}

/* ── Chat messages ─────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* ── Answer metadata ───────────────────────────────────── */
.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 12px;
}
.meta-tile {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
}
.meta-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
}
.meta-value {
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 2px;
}
.chip-row { margin-bottom: 12px; }
.chip {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 500;
    background: var(--brand-soft);
    color: var(--brand);
    border: 1px solid rgba(37, 99, 235, 0.25);
    padding: 3px 10px;
    border-radius: 999px;
    margin: 0 6px 6px 0;
}
.status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 9px 12px;
    border-radius: 10px;
}
.status-ok {
    background: rgba(22, 163, 74, 0.10);
    color: var(--ok);
    border: 1px solid rgba(22, 163, 74, 0.30);
}
.status-warn {
    background: rgba(217, 119, 6, 0.10);
    color: var(--warn);
    border: 1px solid rgba(217, 119, 6, 0.30);
}
.issue-list {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 10px;
}

/* ── Expander / buttons / sidebar ──────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: transparent;
    margin-top: 6px;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid var(--border);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--brand);
    color: var(--brand);
    transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] > button {
    background: var(--brand);
    color: #fff;
    border: none;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--brand-dark);
    color: #fff;
}
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}
.side-brand {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--brand);
    margin-bottom: 2px;
}
.side-tagline {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 4px;
}
.side-topic {
    font-size: 0.88rem;
    padding: 6px 0;
}
.feedback-title {
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Chat input ────────────────────────────────────────── */
[data-testid="stChatInput"] {
    border-radius: 14px;
}

@media (max-width: 640px) {
    .meta-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# 4. SESSION STATE INITIALIZATION

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = set()


# 5. HELPER FUNCTIONS

def render_metadata(metadata):
    """Render the 'Answer details' expander for an assistant response."""

    with st.expander("Answer details"):

        evidence_count = metadata.get("evidence_count", 0)
        latency = metadata.get("latency_seconds", 0)
        tools = metadata.get("tool_calls", []) or []

        st.markdown(
            f"""
<div class="meta-grid">
    <div class="meta-tile">
        <div class="meta-label">Evidence records</div>
        <div class="meta-value">{escape(str(evidence_count))}</div>
    </div>
    <div class="meta-tile">
        <div class="meta-label">Tools used</div>
        <div class="meta-value">{len(tools)}</div>
    </div>
    <div class="meta-tile">
        <div class="meta-label">Response time</div>
        <div class="meta-value">{escape(str(latency))} s</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Tool names as chips
        if tools:
            chips = "".join(
                f'<span class="chip">{escape(str(tool))}</span>'
                for tool in tools
            )
            st.markdown(
                f'<div class="chip-row">{chips}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chip-row"><span class="chip">No tools used</span></div>',
                unsafe_allow_html=True,
            )

        # Validation status
        if metadata.get("validation_passed", False):
            st.markdown(
                '<div class="status status-ok">✅ '
                "Citation and evidence metadata checks passed.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status status-warn">⚠️ '
                "Citation or evidence validation requires attention.</div>",
                unsafe_allow_html=True,
            )

        issues = metadata.get("validation_issues", [])

        if issues:
            items = "".join(f"<li>{escape(str(i))}</li>" for i in issues)
            st.markdown(
                f'<div class="issue-list"><strong>Validation issues</strong>'
                f"<ul>{items}</ul></div>",
                unsafe_allow_html=True,
            )


def submit_feedback(
    message_id,
    question,
    answer,
    rating,
    issue_category,
    comments,
):
    """
    Save student feedback using the feedback manager.

    The feedback manager must support these keyword arguments.
    """

    try:

        result = save_feedback(
            conversation_id=st.session_state.thread_id,
            question=question,
            agent_response=answer,
            rating=rating,
            issue_category=issue_category,
            comments=comments,
        )

        st.session_state.feedback_submitted.add(message_id)

        return True, result

    except Exception as error:

        return False, str(error)


# 6. APPLICATION HEADER

st.markdown(
    """
<div class="hero">
    <div class="hero-badge">Royal University of Bhutan</div>
    <div class="hero-title">🎓 ConductIQ</div>
    <div class="hero-subtitle">
        AI-powered Student Code of Conduct Assistant
    </div>
</div>

<div class="disclaimer">
    <div>ℹ️</div>
    <div>
        <strong>Disclaimer:</strong> ConductIQ provides informational
        guidance based on the RUB Student Code of Conduct 2026.
        It is not an official disciplinary authority. For formal
        clarification, contact your College authority or Student
        Services Officer.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# 7. SIDEBAR

with st.sidebar:

    st.markdown(
        '<div class="side-brand">🎓 ConductIQ</div>'
        '<div class="side-tagline">Your assistant for understanding the '
        "RUB Student Code of Conduct 2026.</div>",
        unsafe_allow_html=True,
    )

    st.write("")

    # Start a new conversation
    if st.button(
        "＋ New conversation",
        use_container_width=True,
        type="primary",
    ):

        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.feedback_submitted = set()

        st.rerun()

    st.divider()

    st.markdown("**What can I help with?**")

    topics = [
        ("⚖️", "Conduct rules and misconduct"),
        ("📋", "Disciplinary procedures"),
        ("🛡️", "Student rights and responsibilities"),
        ("⏱️", "Appeals and deadlines"),
        ("📝", "Academic dishonesty"),
        ("📣", "Reporting student misconduct"),
    ]

    st.markdown(
        "".join(
            f'<div class="side-topic">{icon}&nbsp;&nbsp;{label}</div>'
            for icon, label in topics
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**About ConductIQ**")

    st.caption(
        "Powered by OpenRouter, LangChain and "
        "structured policy retrieval."
    )

    st.caption(
        "Knowledge source: RUB Student Code of Conduct 2026."
    )


# 8. WELCOME SCREEN

if not st.session_state.messages:

    st.markdown(
        '<div class="welcome-title">Welcome to ConductIQ 👋</div>'
        '<div class="welcome-text">Ask questions about the Royal '
        "University of Bhutan Student Code of Conduct 2026.</div>"
        '<div class="example-label">Example questions</div>',
        unsafe_allow_html=True,
    )

    examples = [
        ("📝", "What is academic dishonesty?"),
        ("📣", "How can I report student misconduct?"),
        ("⏱️", "How many working days do I have to appeal a CDC decision?"),
        ("🛡️", "What are my rights during disciplinary proceedings?"),
    ]

    st.markdown(
        "".join(
            f'<div class="example-card"><span class="example-icon">{icon}'
            f"</span><span>{escape(text)}</span></div>"
            for icon, text in examples
        ),
        unsafe_allow_html=True,
    )


# 9. DISPLAY CONVERSATION HISTORY

for index, message in enumerate(st.session_state.messages):

    avatar = "🧑‍🎓" if message["role"] == "user" else "🎓"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

        # Display metadata for assistant responses
        if message["role"] == "assistant":

            metadata = message.get("metadata", {})

            if metadata:
                render_metadata(metadata)


# 10. CHAT INPUT

user_question = st.chat_input(
    "Ask a question about the Student Code of Conduct..."
)


# 11. PROCESS USER QUESTION

if user_question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question,
    })

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_question)

    # Generate agent response
    with st.chat_message("assistant", avatar="🎓"):

        with st.spinner(
            "ConductIQ is retrieving policy evidence..."
        ):

            result = ask_conductiq_detailed(
                question=user_question,
                thread_id=st.session_state.thread_id,
            )

            answer = result["answer"]

        # Display answer
        st.markdown(answer)

        # Collect response metadata
        metadata = {
            "evidence_count": result["evidence_count"],
            "tool_calls": result["tool_calls"],
            "latency_seconds": result["latency_seconds"],
            "validation_passed": result["validation_passed"],
            "validation_issues": result["validation_issues"],
        }

        render_metadata(metadata)

        # Store assistant response
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "metadata": metadata,
            "question": user_question,
        }

        st.session_state.messages.append(
            assistant_message
        )

        st.rerun()


# 12. STUDENT FEEDBACK INTERFACE

if st.session_state.messages:

    last_message = st.session_state.messages[-1]

    if last_message["role"] == "assistant":

        message_id = (
            f"{st.session_state.thread_id}_"
            f"{len(st.session_state.messages)}"
        )

        with st.container(border=True):

            st.markdown(
                '<div class="feedback-title">Was this answer helpful?</div>',
                unsafe_allow_html=True,
            )

            if message_id in st.session_state.feedback_submitted:

                st.success(
                    "Thank you! Your feedback has been recorded."
                )

            else:

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "👍 Helpful",
                        use_container_width=True,
                        key=f"positive_{message_id}",
                    ):

                        success, feedback_result = submit_feedback(
                            message_id=message_id,
                            question=last_message["question"],
                            answer=last_message["content"],
                            rating="positive",
                            issue_category="none",
                            comments="",
                        )

                        if success:

                            st.success(
                                "Thank you for your feedback!"
                            )

                        else:

                            st.error(
                                f"Could not save feedback: {feedback_result}"
                            )

                with col2:

                    if st.button(
                        "👎 Not helpful",
                        use_container_width=True,
                        key=f"negative_{message_id}",
                    ):

                        st.session_state.show_feedback_form = True

                        st.rerun()

                # Additional feedback form
                if st.session_state.get(
                    "show_feedback_form", False
                ):

                    with st.form("negative_feedback_form"):

                        st.markdown(
                            "**What was wrong with the answer?**"
                        )

                        issue_category = st.selectbox(
                            "Issue category",
                            [
                                "incomplete_answer",
                                "incorrect_information",
                                "missing_citation",
                                "irrelevant_answer",
                                "other",
                            ],
                            format_func=lambda v: v.replace("_", " ").capitalize(),
                        )

                        comments = st.text_area(
                            "Additional comments (optional)"
                        )

                        submitted = st.form_submit_button(
                            "Submit feedback",
                            use_container_width=True,
                        )

                        if submitted:

                            success, feedback_result = submit_feedback(
                                message_id=message_id,
                                question=last_message["question"],
                                answer=last_message["content"],
                                rating="negative",
                                issue_category=issue_category,
                                comments=comments,
                            )

                            if success:

                                st.session_state.show_feedback_form = False

                                st.success(
                                    "Your feedback has been submitted "
                                    "for review."
                                )

                                st.rerun()

                            else:

                                st.error(
                                    f"Could not save feedback: {feedback_result}"
                                )