# ui.py: provides an interactive Streamlit dashboard for asking and comparing RAG results

import os
import re

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


# Configure the dashboard title and local API connection controls.
st.set_page_config(page_title="Internal Docs RAG", layout="wide")
st.title("Internal documentation Q&A")
st.caption("Hybrid dense + BM25 retrieval, verified citations, and transparent confidence scores.")
api_url = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip("/")
st.sidebar.caption(f"API endpoint: {api_url}")

# Prefer Streamlit-managed secrets and use deployment environment variables as a fallback.
def _configured_api_key() -> str:
    try:
        return str(st.secrets["api_key"])
    except (KeyError, StreamlitSecretNotFoundError):
        return os.getenv("STREAMLIT_API_KEY", "")


# Initialize per-user conversation and result state in one predictable place.
for state_key, default in {
    "thread_id": "dashboard-session",
    "last_result": None,
    "last_compare": False,
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default

api_key = st.sidebar.text_input("API key", value=_configured_api_key(), type="password")


# Send a JSON request with the configured API key and report HTTP failures explicitly.
def _post(path: str, payload: dict):
    headers = {"X-API-Key": api_key} if api_key else {}
    response = requests.post(
        f"{api_url}{path}", json=payload, headers=headers, timeout=(5, 180)
    )
    response.raise_for_status()
    return response.json()


# Batch question entry and retrieval mode selection into one form submission.
with st.form("ask_form"):
    question = st.text_area("Ask a question", placeholder="What is the incident escalation process?")
    compare = st.toggle("Compare hybrid retrieval with dense-only retrieval")
    submitted = st.form_submit_button(
        "Ask", type="primary", disabled=not question.strip(), width="stretch"
    )

# Run the selected retrieval workflow and show explicit API errors.
if submitted:
    payload = {
        "question": question.strip(),
        "retrieval_mode": "hybrid",
        "thread_id": st.session_state.thread_id,
    }
    try:
        response = _post("/v1/compare" if compare else "/v1/ask", payload)
        st.session_state.last_result = response
        st.session_state.last_compare = compare
    except requests.RequestException as error:
        st.error(f"Could not reach the RAG API: {error}")
        if error.response is not None:
            st.code(error.response.text)


# Render one answer together with confidence, evidence, and citation-verification results.
def _render_answer(result: dict, title: str):
    st.subheader(title)
    anchor_prefix = re.sub(r"[^a-z0-9-]", "", title.casefold().replace(" ", "-"))
    linked_answer = re.sub(
        r"\[(\d+)\]",
        lambda match: f'[{match.group(1)}](#{anchor_prefix}-chunk-{match.group(1)})',
        result["answer"],
    )
    st.markdown(linked_answer)
    st.metric("Overall confidence", f'{result["confidence"]:.0%}')
    st.json(result["confidence_breakdown"])
    if result["insufficient_context"]:
        st.warning("Low retrieval confidence: treat this as an explicit no-answer.")
        st.write("Documents to check:", ", ".join(result["documents_to_check"]) or "None indexed")
    with st.container(border=True):
        st.subheader("Retrieved chunks and scores")
        for item in result["retrieved_chunks"]:
            chunk = item["chunk"]
            location = f'{chunk["source"]} - {chunk.get("section_heading") or "No section"}'
            if chunk.get("page_number"):
                location += f' - page {chunk["page_number"]}'
            st.header(
                f'[{item["rank"]}] {location}',
                anchor=f'{anchor_prefix}-chunk-{item["rank"]}',
            )
            st.caption(
                f'Dense: {item["dense_score"]} - BM25: {item["bm25_score"]} - '
                f'Fusion: {item["rrf_score"]} - Rerank: {item["rerank_score"]}'
            )
            st.write(chunk["text"])
    with st.expander("Citation verification"):
        if not result["citations"]:
            st.info("No answer claims were generated for citation checking.")
        for citation in result["citations"]:
            status = "Verified" if citation["verified"] else "Unsupported"
            st.write(
                f'{status} - [{citation["citation_number"]}] '
                f'{citation["source"] or "No source"}: {citation["claim"]}'
            )
            st.caption(citation["rationale"])


# Display the latest answer or both retrieval modes side by side.
if st.session_state.last_result is not None:
    saved = st.session_state.last_result
    if st.session_state.last_compare:
        left, right = st.columns(2)
        with left:
            _render_answer(saved["hybrid"], "Hybrid")
        with right:
            _render_answer(saved["dense"], "Dense only")
    else:
        _render_answer(saved, "Answer")

# Let reviewers inspect the documents currently present in the local index.
if st.sidebar.button("Refresh indexed documents"):
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        response = requests.get(f"{api_url}/v1/documents", headers=headers, timeout=15)
        response.raise_for_status()
        st.sidebar.json(response.json())
    except requests.RequestException as error:
        st.sidebar.error(f"Could not load document inventory: {error}")
