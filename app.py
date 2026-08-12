import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.graph import workflow

# --- Page setup -------------------------------------------------------------
st.set_page_config(
    page_title="Console // CRAG",
    page_icon="🜂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Visual identity: instrument-panel dark theme, amber accent ------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #14161A; color: #E8E6E1; }
    .stApp p, .stApp li, [data-testid="stMarkdownContainer"] p {
        color: #E8E6E1 !important;
    }

    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

    section[data-testid="stSidebar"] {
        background-color: #1C1F26;
        border-right: 1px solid #2A2E37;
    }
    section[data-testid="stSidebar"] label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #9AA1AC;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #262B33;
        border-radius: 10px;
        padding: 0.35rem 0.75rem;
        margin-bottom: 0.4rem;
        background-color: #1A1D23;
    }

    /* the chat input keeps Streamlit's light input surface; darken the text to match */
    div[data-testid="stChatInput"] textarea {
        color: #14161A !important;
        caret-color: #14161A;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #5B6472 !important;
        opacity: 1;
    }

    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background-color: #4ADE80;
        margin-right: 6px;
        box-shadow: 0 0 6px #4ADE80;
    }
    .status-dot.off { background-color: #6B7280; box-shadow: none; }

    .step-caption {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #E8A33D;
    }

    div.stButton > button {
        border: 1px solid #E8A33D;
        color: #E8A33D;
        background-color: transparent;
        font-family: 'JetBrains Mono', monospace;
    }
    div.stButton > button:hover {
        background-color: #E8A33D;
        color: #14161A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Console // CRAG")
st.caption(
    "Corrective RAG over agents, prompt engineering, and adversarial attacks on LLMs. "
    "Anything else is routed to the web."
)


def _format_sources(context_docs: List[Any]) -> List[str]:
    sources = [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]
    return list(dict.fromkeys(sources))


STEP_LABELS = {
    RETRIEVE: "retrieving from the vector store",
    GRADE_DOCUMENTS: "grading document relevance",
    WEB_SEARCH: "searching the web",
    GENERATE: "generating answer",
}

def _get_secret(key: str) -> str | None:
    # st.secrets raises (rather than returning a default) when no secrets.toml exists at all
    try:
        return st.secrets.get(key)
    except Exception:
        return None


# --- Key check: the graph needs both OpenAI (LLMs) and Tavily (web search) -
openai_key = _get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
tavily_key = _get_secret("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

if not openai_key or not tavily_key:
    missing = ", ".join(
        name for name, val in [("OPENAI_API_KEY", openai_key), ("TAVILY_API_KEY", tavily_key)] if not val
    )
    st.warning(
        f"Missing {missing}. Add it to `.streamlit/secrets.toml` or your `.env` file, "
        "then rerun the app.",
        icon="⚠️",
    )

keys_ready = bool(openai_key and tavily_key)

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    dot_class = "status-dot" if keys_ready else "status-dot off"
    st.markdown(
        f'<span class="{dot_class}"></span>**{"ready" if keys_ready else "keys missing"}**',
        unsafe_allow_html=True,
    )
    st.markdown("### Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Session state: chat history persists across reruns --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about AI agents. I will retrieve context and cite sources.",
            "sources": [],
        }
    ]

# Re-render every prior bubble so history survives a script rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"- {s}")

# --- Chat input --------------------------------------------------------------
prompt = st.chat_input("Ask a question about AI agents...", disabled=not keys_ready)

if prompt:
    # Append the user turn immediately so the UI reflects it before the graph runs
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        placeholder = st.empty()
        final_state: Dict[str, Any] = {}
        answer = ""
        sources: List[str] = []
        try:
            # Stream the graph node by node, surfacing each step as it runs
            for step in workflow.stream({"question": prompt}):
                for node_name, update in step.items():
                    status.markdown(
                        f'<span class="step-caption">→ {STEP_LABELS.get(node_name, node_name)}…</span>',
                        unsafe_allow_html=True,
                    )
                    final_state.update(update)
            status.empty()

            answer = final_state.get("generation", "")
            sources = _format_sources(final_state.get("documents", []))
            placeholder.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")
        except Exception as e:
            status.empty()
            answer = "Failed to generate a response."
            placeholder.error(answer)
            st.exception(e)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
