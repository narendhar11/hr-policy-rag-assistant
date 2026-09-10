"""
HR Policy RAG Assistant — Streamlit UI
Vertexon Solutions | Powered by LangChain + Groq + FAISS + Jina Embeddings
"""

import sys
import os
import streamlit as st

from pipeline import build_hr_policy_assistant, ask_hr_policy_question
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant | Vertexon Solutions",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — works WITH Streamlit dark theme, not against it ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Main area */
.block-container {
    padding: 1.5rem 2rem 5rem 2rem !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155;
}

/* Hero */
.hero { margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B; }
.hero h1 { font-size: 1.6rem; font-weight: 600; color: #F1F5F9; margin: 0 0 0.25rem 0; }
.hero h1 em { font-style: normal; color: #10B981; }
.hero p { color: #64748B; font-size: 0.875rem; margin: 0; }

/* Sidebar brand */
.sb-brand { padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid #334155; }
.sb-logo {
    width: 36px; height: 36px; background: #10B981;
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 1.1rem; margin-bottom: 0.6rem;
}
.sb-name { font-size: 0.95rem; font-weight: 600; color: #F1F5F9; }
.sb-role { font-size: 0.75rem; color: #64748B; margin-top: 0.1rem; }

/* Status */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 0.25rem 0.65rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 500;
}
.badge-ready  { background: #064E3B; color: #6EE7B7; border: 1px solid #065F46; }
.badge-load   { background: #1C1917; color: #FCD34D; border: 1px solid #78350F; }
.badge-error  { background: #450A0A; color: #FCA5A5; border: 1px solid #7F1D1D; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* Stats */
.stats { display: flex; gap: 0.6rem; margin: 1rem 0; }
.stat {
    flex: 1; background: #0F172A;
    border: 1px solid #334155; border-radius: 6px; padding: 0.6rem 0.75rem;
}
.stat-n { font-size: 1.3rem; font-weight: 600; color: #10B981; }
.stat-l { font-size: 0.7rem; color: #64748B; margin-top: 0.15rem; }

.divider { height: 1px; background: #334155; margin: 0.75rem 0; }

/* Stack */
.stack-h { font-size: 0.7rem; color: #475569; margin-bottom: 0.5rem; font-weight: 500; }
.stack-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.35rem 0; border-bottom: 1px solid #1E293B;
}
.stack-row:last-child { border-bottom: none; }
.stack-n { font-size: 0.8rem; font-weight: 500; color: #CBD5E1; }
.stack-r { font-size: 0.68rem; color: #475569; }
.pip { width: 5px; height: 5px; border-radius: 50%; background: #334155; flex-shrink: 0; }

/* Sample question buttons */
div[data-testid="stButton"] > button {
    background: #1E293B !important;
    color: #94A3B8 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transition: border-color 0.15s, color 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #10B981 !important;
    color: #10B981 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Bottom bar */
[data-testid="stBottom"] {
    background: #0F172A !important;
    border-top: 1px solid #1E293B;
    padding: 0.75rem 1.5rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }

/* Chat messages */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] { animation: fadeUp 0.15s ease; }

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
}

/* Assistant bubble — emerald left accent */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-left: 2px solid #10B981;
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
}

/* Avatar squares */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 28px !important; height: 28px !important;
    min-width: 28px !important; border-radius: 6px !important;
}
[data-testid="stChatMessageAvatarUser"]      { background: #334155 !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: #064E3B !important; border: 1px solid #10B981 !important; }

/* Tables inside chat */
[data-testid="stChatMessage"] table {
    border-collapse: collapse; width: 100%; font-size: 0.85rem; margin: 0.5rem 0;
}
[data-testid="stChatMessage"] th {
    background: #1E293B; padding: 0.4rem 0.7rem;
    border: 1px solid #334155; text-align: left; font-weight: 600;
}
[data-testid="stChatMessage"] td {
    padding: 0.35rem 0.7rem; border: 1px solid #1E293B; line-height: 1.5;
}
[data-testid="stChatMessage"] tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

/* Footer */
.footer {
    position: fixed; bottom: 0; left: 0; right: 0;
    text-align: center; padding: 0.3rem;
    font-size: 0.65rem; color: #334155;
    background: #0F172A; border-top: 1px solid #1E293B;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages"        not in st.session_state: st.session_state.messages = []
if "agent"           not in st.session_state: st.session_state.agent = None
if "agent_status"    not in st.session_state: st.session_state.agent_status = "idle"
if "total_questions" not in st.session_state: st.session_state.total_questions = 0
if "pending_question"not in st.session_state: st.session_state.pending_question = None


# ── Load agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    """Load and cache the HR policy agent (runs once per session)."""
    return build_hr_policy_assistant()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    status = st.session_state.agent_status
    badge = {
        "ready":   '<span class="badge badge-ready"><span class="dot"></span>Ready</span>',
        "error":   '<span class="badge badge-error"><span class="dot"></span>Error</span>',
    }.get(status, '<span class="badge badge-load"><span class="dot"></span>Loading…</span>')

    q  = st.session_state.total_questions
    ch = len(st.session_state.messages) // 2

    st.markdown(f"""
    <div class="sb-brand">
        <div class="sb-logo">🏢</div>
        <div class="sb-name">Vertexon Solutions</div>
        <div class="sb-role">HR Policy Assistant</div>
    </div>
    {badge}
    <div class="stats">
        <div class="stat"><div class="stat-n">{q}</div><div class="stat-l">Questions asked</div></div>
        <div class="stat"><div class="stat-n">{ch}</div><div class="stat-l">Exchanges</div></div>
    </div>
    <div class="divider"></div>
    <div class="stack-h">Powered by</div>
    <div class="stack-row"><div class="pip"></div><div><div class="stack-n">LangChain</div><div class="stack-r">Agent framework</div></div></div>
    <div class="stack-row"><div class="pip"></div><div><div class="stack-n">Groq</div><div class="stack-r">LLM inference</div></div></div>
    <div class="stack-row"><div class="pip"></div><div><div class="stack-n">FAISS</div><div class="stack-r">Vector store</div></div></div>
    <div class="stack-row"><div class="pip"></div><div><div class="stack-n">Jina</div><div class="stack-r">Embeddings</div></div></div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>HR Policy <em>Assistant</em></h1>
    <p>Ask anything about Vertexon Solutions policies — leave, benefits, remote work, and more.</p>
</div>
""", unsafe_allow_html=True)


# ── Initialize agent ──────────────────────────────────────────────────────────
if st.session_state.agent is None and st.session_state.agent_status != "error":
    st.session_state.agent_status = "loading"
    with st.spinner("Loading knowledge base…"):
        try:
            st.session_state.agent = load_agent()
            st.session_state.agent_status = "ready"
            logger.info("HR Policy Assistant initialized successfully.")
        except Exception as e:
            st.session_state.agent_status = "error"
            logger.error("Failed to initialize HR Policy Assistant: %s", e, exc_info=True)
            st.error(f"Could not load the assistant: {e}")
            st.stop()
    st.rerun()


# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is the remote work policy?",
    "How do I apply for leave?",
]

if not st.session_state.messages:
    st.caption("Try one of these to get started")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, (col, q) in enumerate(zip(cols, SAMPLE_QUESTIONS)):
        with col:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about leave, benefits, remote work, or any HR policy…")


# ── Resolve question source ───────────────────────────────────────────────────
question_to_ask = None
if st.session_state.pending_question:
    question_to_ask = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_input and user_input.strip():
    question_to_ask = user_input.strip()


# ── Handle question ───────────────────────────────────────────────────────────
if question_to_ask and st.session_state.agent:
    logger.info("User asked: '%s'", question_to_ask)
    st.session_state.messages.append({"role": "user", "content": question_to_ask})
    st.session_state.total_questions += 1

    with st.chat_message("user", avatar="👤"):
        st.markdown(question_to_ask)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching policies…"):
            try:
                answer = ask_hr_policy_question(st.session_state.agent, question_to_ask)
            except Exception as e:
                logger.error("Error answering question '%s': %s", question_to_ask, e, exc_info=True)
                answer = f"Something went wrong while searching. Please try again.\n\n`{str(e)}`"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Vertexon Solutions · HR Policy Assistant</div>',
    unsafe_allow_html=True
)
