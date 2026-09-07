"""
HR Policy RAG Assistant — Streamlit UI
Vertexon Solutions | Powered by LangChain + Groq + FAISS + Jina Embeddings
"""

import sys
import os
import streamlit as st
import time

from pipeline import build_hr_policy_assistant, ask_hr_policy_question

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant | Vertexon Solutions",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main block: centered, proper padding ── */
.block-container {
    padding: 0 2rem 5rem 2rem !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* ── Sidebar — default width, tighter inner padding ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    height: 100vh;
    overflow-y: auto;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0.3rem 1rem 1rem 1rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.3rem !important; }
[data-testid="stSidebar"] .stat-card { margin-bottom: 0.3rem !important; padding: 0.6rem 0.8rem !important; }
[data-testid="stSidebar"] .glass-divider { margin: 0.4rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin: 0 !important; }


/* ── Hero header (pinned top) ── */
.hero-header {
    text-align: center;
    padding: 1.2rem 1rem 0.8rem;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(15, 12, 41, 0.6);
    backdrop-filter: blur(12px);
}
.hero-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.hero-header p {
    color: rgba(255,255,255,0.45);
    font-size: 0.85rem;
    font-weight: 400;
    margin: 0;
}

/* ── Scrollable chat messages area ── */
#chat-scroll-area {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 1.5rem;
    scroll-behavior: smooth;
}
#chat-scroll-area::-webkit-scrollbar { width: 4px; }
#chat-scroll-area::-webkit-scrollbar-track { background: transparent; }
#chat-scroll-area::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
}

/* ── Sample question pills ── */
.sample-pills {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    padding: 1.5rem 0 0.5rem;
}

/* ── Stat cards (sidebar) ── */
.stat-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.3rem 0.8rem;
    border-radius: 20px; font-size: 0.78rem; font-weight: 500;
}
.status-badge.ready {
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34d399;
}
.status-badge.loading {
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid rgba(251, 191, 36, 0.3);
    color: #fbbf24;
}
.status-badge.error {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* ── Divider ── */
.glass-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    margin: 1rem 0;
}

/* ── Animations ── */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%           { transform: translateY(-6px); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}

/* ── Chat input (sticky bottom) ── */
[data-testid="stBottom"] {
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(16px);
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 0.8rem 1.5rem !important;
    flex-shrink: 0;
}
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(255,255,255,0.28) !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(99, 102, 241, 0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 18px rgba(99, 102, 241, 0.5) !important;
}

/* ── Scrollbar (global) ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = "idle"   # idle | loading | ready | error
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Load agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    """Load and cache the HR policy agent (runs once per session)."""
    return build_hr_policy_assistant()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    status = st.session_state.agent_status
    if status == "ready":
        status_html = '<div class="status-badge ready"><div class="status-dot"></div> Assistant Ready</div>'
    elif status == "error":
        status_html = '<div class="status-badge error"><div class="status-dot"></div> Error</div>'
    else:
        status_html = '<div class="status-badge loading"><div class="status-dot"></div> Initializing…</div>'

    q_count = st.session_state.total_questions
    chat_count = len(st.session_state.messages) // 2

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:0.6rem; padding-top:0.2rem;">
        <div style="text-align:center; padding:0.5rem 0;">
            <div style="font-size:2.4rem; line-height:1;">🏢</div>
            <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-top:0.25rem;">Vertexon Solutions</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:0.1rem;">HR Policy Assistant</div>
        </div>
        <div class="glass-divider"></div>
        {status_html}
        <div class="glass-divider"></div>
        <div style="display:flex; gap:0.4rem;">
            <div class="stat-card" style="flex:1; padding:0.5rem; margin:0;">
                <div class="stat-value" style="font-size:1.3rem;">{q_count}</div>
                <div class="stat-label">Questions</div>
            </div>
            <div class="stat-card" style="flex:1; padding:0.5rem; margin:0;">
                <div class="stat-value" style="font-size:1.3rem;">{chat_count}</div>
                <div class="stat-label">Chats</div>
            </div>
        </div>
        <div class="glass-divider"></div>
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.07em;">Tech Stack</div>
        <div style="display:flex; flex-direction:column; gap:0.35rem;">
            <div style="display:flex; align-items:center; gap:8px; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>🔗</span>
                <div>
                    <div style="font-size:0.8rem; font-weight:600; color:#e2e8f0;">LangChain</div>
                    <div style="font-size:0.68rem; color:rgba(255,255,255,0.35);">Agent Framework</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:8px; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>⚡</span>
                <div>
                    <div style="font-size:0.8rem; font-weight:600; color:#e2e8f0;">Groq</div>
                    <div style="font-size:0.68rem; color:rgba(255,255,255,0.35);">LLM Inference</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:8px; padding:0.3rem 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>🗂️</span>
                <div>
                    <div style="font-size:0.8rem; font-weight:600; color:#e2e8f0;">FAISS</div>
                    <div style="font-size:0.68rem; color:rgba(255,255,255,0.35);">Vector Store</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:8px; padding:0.3rem 0;">
                <span>🔍</span>
                <div>
                    <div style="font-size:0.8rem; font-weight:600; color:#e2e8f0;">Jina</div>
                    <div style="font-size:0.68rem; color:rgba(255,255,255,0.35);">Embeddings</div>
                </div>
            </div>
        </div>
        <div class="glass-divider"></div>
    </div>
    """, unsafe_allow_html=True)


    # Clear chat button (must be native Streamlit widget)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.rerun()


# ── Main content ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏢 HR Policy Assistant</h1>
    <p>Ask anything about Vertexon Solutions policies — powered by AI</p>
</div>
""", unsafe_allow_html=True)

# ── Initialize agent on first load ────────────────────────────────────────────
if st.session_state.agent is None and st.session_state.agent_status != "error":
    st.session_state.agent_status = "loading"
    with st.spinner(""):
        st.markdown("""
        <div style="text-align:center; padding: 2rem; color: rgba(255,255,255,0.5);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem; animation: pulse 1.5s infinite;">⚙️</div>
            <div style="font-size: 0.9rem;">Initializing HR Policy Assistant…</div>
            <div style="font-size: 0.78rem; margin-top: 0.3rem; color: rgba(255,255,255,0.3);">
                Loading knowledge base & building vector store
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            st.session_state.agent = load_agent()
            st.session_state.agent_status = "ready"
        except Exception as e:
            st.session_state.agent_status = "error"
            st.error(f"❌ Failed to initialize assistant: {e}")
            st.stop()
    st.rerun()


# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is the remote work policy?",
    "How do I apply for leave?",
]

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; margin-bottom: 1rem;">
        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.35); text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.8rem;">
            Try asking…
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, (col, q) in enumerate(zip(cols, SAMPLE_QUESTIONS)):
        with col:
            if st.button(q, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# ── Extra CSS: st.chat_message styling + scrollable area ─────────────────────
st.markdown("""
<style>
/* ── Scrollable chat history wrapper ── */
[data-testid="stVerticalBlockBorderWrapper"]:has(#chat-scroll-sentinel) > div {
    max-height: calc(100vh - 230px);
    overflow-y: auto;
    padding: 0.5rem 0;
    scroll-behavior: smooth;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
    background: rgba(99, 102, 241, 0.09);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0 0.4rem 3rem;
    animation: slideInRight 0.3s ease;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: #e2e8f0;
    margin: 0;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 4px 16px 16px 16px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 3rem 0.4rem 0;
    animation: slideInLeft 0.3s ease;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) li,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) td,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) th {
    color: #e2e8f0 !important;
}

/* Tables */
[data-testid="stChatMessage"] table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6rem 0;
    font-size: 0.87rem;
}
[data-testid="stChatMessage"] th {
    background: rgba(99, 102, 241, 0.18);
    padding: 0.45rem 0.75rem;
    text-align: left;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.1);
}
[data-testid="stChatMessage"] td {
    padding: 0.4rem 0.75rem;
    border: 1px solid rgba(255,255,255,0.07);
    vertical-align: top;
    line-height: 1.5;
}
[data-testid="stChatMessage"] tr:nth-child(even) td {
    background: rgba(255,255,255,0.03);
}

/* Avatars */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    border-radius: 50% !important;
    font-size: 1rem !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #34d399, #059669) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Scrollable chat area ───────────────────────────────────────────────────────
with st.container():
    # Sentinel div — CSS targets the container that holds this
    st.markdown('<div id="chat-scroll-sentinel"></div>', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# Auto-scroll to bottom after new messages
if st.session_state.messages:
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)
    st.markdown("""
    <script>
        const el = document.getElementById('chat-bottom');
        if (el) el.scrollIntoView({behavior: 'smooth'});
    </script>
    """, unsafe_allow_html=True)


# ── Chat input (sticky at bottom via stBottom) ────────────────────────────────
user_input = st.chat_input("Ask about HR policies, leave, benefits, remote work…")


# ── Handle pending question (from sample buttons) ─────────────────────────────
question_to_ask = None
if st.session_state.pending_question:
    question_to_ask = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_input and user_input.strip():
    question_to_ask = user_input.strip()


# ── Process question ───────────────────────────────────────────────────────────
if question_to_ask and st.session_state.agent:
    st.session_state.messages.append({"role": "user", "content": question_to_ask})
    st.session_state.total_questions += 1
    with st.chat_message("user", avatar="👤"):
        st.markdown(question_to_ask)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching HR policies…"):
            try:
                answer = ask_hr_policy_question(st.session_state.agent, question_to_ask)
            except Exception as e:
                answer = f"⚠️ Sorry, I encountered an error: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ── Footer (fixed at bottom of viewport) ──────────────────────────────────────
st.markdown("""
<style>
.app-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    text-align: center;
    padding: 0.4rem 0;
    color: rgba(255,255,255,0.18);
    font-size: 0.7rem;
    background: rgba(15, 12, 41, 0.75);
    backdrop-filter: blur(10px);
    border-top: 1px solid rgba(255,255,255,0.05);
    z-index: 9999;
    letter-spacing: 0.03em;
}
</style>
<div class="app-footer">
    HR Policy Assistant &nbsp;·&nbsp; Vertexon Solutions &nbsp;·&nbsp; Powered by LangChain &amp; Groq
</div>
""", unsafe_allow_html=True)
