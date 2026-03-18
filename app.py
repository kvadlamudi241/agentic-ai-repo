import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Web Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 1.4rem;
}
.app-header .title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 500;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.app-header .subtitle {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 2px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Status pill ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 1.2rem;
    font-size: 0.82rem;
    font-family: 'IBM Plex Mono', monospace;
}
.status-row.active {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #86efac;
}
.status-row.inactive {
    background: rgba(148, 163, 184, 0.07);
    border: 1px solid rgba(148, 163, 184, 0.15);
    color: #94a3b8;
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.status-dot.green { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.status-dot.gray  { background: #64748b; }

/* ── Tool badge ── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 4px;
    font-size: 0.73rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.tool-badge.on {
    background: rgba(99, 179, 237, 0.12);
    border: 1px solid rgba(99, 179, 237, 0.25);
    color: #90cdf4;
}
.tool-badge.off {
    background: rgba(100, 116, 139, 0.1);
    border: 1px solid rgba(100, 116, 139, 0.2);
    color: #64748b;
    text-decoration: line-through;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    background: #151820 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
    color: #e2e8f0 !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #151820 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Sidebar text input ── */
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: #0d0f14 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 7px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #3b82f6 !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label span {
    font-size: 0.88rem !important;
    color: #cbd5e1 !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}

/* ── Selectbox ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0d0f14 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 7px !important;
    color: #e2e8f0 !important;
}

/* ── Divider ── */
hr {
    border-color: #1e2330 !important;
    margin: 1.2rem 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    color: #3b82f6 !important;
}

/* ── Error / warning ── */
[data-testid="stAlert"] {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    border-radius: 8px !important;
    color: #fca5a5 !important;
    font-size: 0.85rem !important;
}

/* ── Sidebar section header ── */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.4rem 0 0.5rem;
}

/* ── Info box ── */
.info-box {
    background: rgba(59, 130, 246, 0.07);
    border: 1px solid rgba(59, 130, 246, 0.18);
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    font-size: 0.8rem;
    color: #93c5fd;
    line-height: 1.55;
    margin-top: 0.8rem;
}

/* ── Button (clear) ── */
[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid #1e2330 !important;
    color: #64748b !important;
    font-size: 0.78rem !important;
    border-radius: 7px !important;
    width: 100% !important;
    padding: 0.4rem 0.75rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #334155 !important;
    color: #94a3b8 !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── Metric ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1rem;
}
.stat-card {
    flex: 1;
    background: #151820;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.stat-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 500;
    color: #e2e8f0;
}
.stat-card .lbl {
    font-size: 0.68rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 2px;
}

/* ── Welcome screen ── */
.welcome {
    text-align: center;
    padding: 4rem 1rem;
    color: #475569;
}
.welcome h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: #64748b;
    margin-bottom: 0.5rem;
    font-weight: 400;
}
.welcome p { font-size: 0.85rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE INITIALISATION ───────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "msg_count"    not in st.session_state: st.session_state.msg_count    = 0
if "search_count" not in st.session_state: st.session_state.search_count = 0


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-header"><div><div class="title">⬡ AI Agent</div><div class="subtitle">LangGraph · Groq · DuckDuckGo</div></div></div>', unsafe_allow_html=True)

    # ── Credentials ──
    st.markdown('<div class="sidebar-section">Credentials</div>', unsafe_allow_html=True)
    user_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    # ── Model ──
    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview"],
        label_visibility="collapsed",
    )

    # ── Tool Toggle ──────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Tools</div>', unsafe_allow_html=True)
    use_search = st.toggle("Enable Web Search", value=True)

    if use_search:
        st.markdown(
            '<span class="tool-badge on">🌐 duckduckgo_search &nbsp;ON</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="tool-badge off">🌐 duckduckgo_search &nbsp;OFF</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="info-box">When <b>OFF</b>, the agent answers from its training knowledge only — faster and uses fewer API tokens.</div>', unsafe_allow_html=True)

    # ── System Prompt ──
    st.markdown('<div class="sidebar-section">System Prompt</div>', unsafe_allow_html=True)
    system_prompt = st.text_area(
        "Instructions",
        value="You are a Senior research assistant. When web search is enabled, always use it to find current information before answering. When it is disabled, answer from your training knowledge and clearly say so.",
        height=120,
        label_visibility="collapsed",
    )

    # ── Stats ──
    st.markdown('<div class="sidebar-section">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="val">{st.session_state.msg_count}</div><div class="lbl">Messages</div></div>
      <div class="stat-card"><div class="val">{st.session_state.search_count}</div><div class="lbl">Searches</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Clear ──
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages     = []
        st.session_state.msg_count    = 0
        st.session_state.search_count = 0
        st.rerun()


# ─── MAIN PANEL ─────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.25rem;font-weight:500;color:#e2e8f0;padding-top:0.2rem">AI Web Agent</div>', unsafe_allow_html=True)
with col_status:
    if use_search:
        st.markdown('<div class="status-row active"><div class="status-dot green"></div>Web search active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-row inactive"><div class="status-dot gray"></div>Knowledge mode</div>', unsafe_allow_html=True)

st.markdown("---")

# Render existing chat history
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
      <h2>Ready to assist</h2>
      <p>Ask me anything. Toggle <b>Enable Web Search</b> in the sidebar<br>
      to switch between live internet access and training knowledge.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ─── CHAT INPUT & AGENT LOOP ────────────────────────────────────────────────
if user_query := st.chat_input("Ask me anything…"):

    # ── Guard: need API key ──
    if not user_api_key:
        st.error("Please enter your Groq API key in the sidebar to continue.")
        st.stop()

    # ── Show & save user message ──
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # ── 4A: Initialise the LLM ──
    llm = ChatGroq(
        model=model_choice,
        temperature=0,
        api_key=user_api_key,
    )

    # ── 4B: THE TOOL TOGGLE LOGIC — web_tool only created when needed ───────
    if use_search:
        web_tool = DuckDuckGoSearchRun(
            name="duckduckgo_search",
            description="Search the internet for current events and up-to-date information.",
        )
        active_tools = [web_tool]
    else:
        active_tools = []

    # ── 4C: Build agent with active_tools ──────────────────────────────────
    agent = create_react_agent(llm, tools=active_tools)

    # ── 4D: Build LangGraph-format history (bridge pattern) ─────────────────
    langgraph_history = [SystemMessage(content=system_prompt)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            langgraph_history.append(HumanMessage(content=m["content"]))
        else:
            langgraph_history.append(AIMessage(content=m["content"]))

    # ── 4E: Run the agent ──
    with st.chat_message("assistant"):
        spinner_label = (
            "🌐 Searching the web and reasoning…"
            if use_search
            else "🧠 Reasoning from training knowledge…"
        )
        with st.spinner(spinner_label):
            try:
                result_state = agent.invoke({"messages": langgraph_history})
                bot_answer   = result_state["messages"][-1].content

                # Count searches made (check for ToolMessage in result)
                tool_msgs = [
                    m for m in result_state["messages"]
                    if hasattr(m, "type") and m.type == "tool"
                ]
                if tool_msgs:
                    st.session_state.search_count += len(tool_msgs)

            except Exception as e:
                bot_answer = f"⚠️ Agent error: {str(e)}"

        st.markdown(bot_answer)

    # ── Save assistant reply & update counter ──
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
    st.session_state.msg_count += 1
    st.rerun()
