import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", page_icon="📚")
st.title("📚 Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = None


def show_metrics(box, m):
    with box.container():
        st.subheader("Last query")
        if not m:
            st.caption("No queries yet.")
            return
        st.metric("Latency", f"{m['latency_ms']} ms")
        st.metric("Cost", f"${m['cost_usd']:.5f}")
        st.write(f"Tokens in/out: {m['prompt_tokens']} / {m['completion_tokens']}")
        st.write("Served from cache ✅" if m["cached"] else "Fresh answer")


def show_citations(citations):
    for c in citations:
        with st.expander(f"{c['doc']}: {c['source_file']} ({c['section']})"):
            st.write(c["text"])


# ---------- sidebar ----------
with st.sidebar:
    st.header("Upload a document")
    up = st.file_uploader("PDF, DOCX, HTML or Markdown", type=["pdf", "docx", "html", "htm", "md"])
    if up and st.button("Index it"):
        with st.spinner("Indexing..."):
            r = requests.post(f"{API_URL}/upload", files={"file": (up.name, up.getvalue())}, timeout=600)
        if r.ok:
            st.success(f"Added {r.json()['chunks_added']} chunks from {up.name}")
        else:
            st.error(r.text)

    st.header("Filter")
    file_type = st.selectbox("File type", ["any", "pdf", "docx", "html", "md"])
    metrics_box = st.empty()

show_metrics(metrics_box, st.session_state.metrics)

# ---------- chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            show_citations(msg["citations"])

# ---------- new question ----------
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    payload = {"question": question}
    if file_type != "any":
        payload["filters"] = {"file_type": file_type}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                data = None
                st.error(f"API error: {e}")
        if data:
            st.write(data["answer"])
            show_citations(data["citations"])
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "citations": data["citations"]}
            )
            st.session_state.metrics = data
            show_metrics(metrics_box, data)