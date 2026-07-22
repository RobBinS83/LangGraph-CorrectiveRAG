from typing import Any, Dict, List
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

from graph.graph import workflow

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta:= (getattr(doc, "metadata", None) or {})) is not None
    ]

st.set_page_config(page_title="LangGraph Documentation Helper", layout="centered")
st.title("LangGraph Documentation Helper")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about AI agents. I will retrieve context and cite sources.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about AI agent...")
if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": []
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answers..."):
                result = workflow.invoke(input={"question": prompt})

            answer = result.get("generation", "")
            sources = _format_sources(result.get("documents", []))

            st.markdown(answer)
            if sources:
                with st.expander("sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as e:
            st.error("Failed to generate a response")
            st.exception(e)


#if __name__ == "__main__":
#    print("Hello from Corrective-RAG!")
#    print(workflow.invoke(input={"question": "How to make pizza?"}))
