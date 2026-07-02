"""
Streamlit frontend for RAG Knowledge Base.
- Upload files (PDF/TXT) via sidebar and auto-ingest
- Query box for natural language QA
- Shows green answer box, confidence warning, expandable sources, and doc count
"""

import os
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Knowledge Base", layout="wide")
st.title(" RAG Knowledge Base")

# Sidebar for uploads
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if uploaded_file:
    data_folder = os.getenv("DATA_PATH", "./data")
    os.makedirs(data_folder, exist_ok=True)
    save_path = os.path.join(data_folder, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")
    with st.spinner("Ingesting..."):
        res = requests.post(f"{API_URL}/ingest")
        if res.status_code == 200:
            st.sidebar.success("Document ingested & vector store updated.")
        else:
            st.sidebar.error(f"Ingestion failed: {res.json().get('detail', 'Error.')}")

# Sidebar: doc count
def get_docs_count():
    try:
        r = requests.get(f"{API_URL}/health")
        return r.json().get("doc_count", "?")
    except Exception:
        return "?"

st.sidebar.markdown(f"**Total documents in knowledge base:** {get_docs_count()}")

st.markdown("### Ask a Question")
question = st.text_input("Enter your question...", "")
submit = st.button("Submit Question")

if submit and not question.strip():
    st.warning("Please enter a question before submitting.")

if submit and question.strip():
    with st.spinner("Getting answer..."):
        try:
            r = requests.post(f"{API_URL}/query", json={"question": question})
            if r.status_code == 200:
                result = r.json()
                answer = result["answer"]
                confidence = result.get("confidence", 0.0)
                sources = result.get("sources", [])
                box_color = "#d2ffd6"
                st.markdown(
                    f'<div style="background-color:{box_color};padding:1em;border-radius:7px;margin-bottom:10px">'
                    f"<b>Answer:</b><br>{answer}</div>", unsafe_allow_html=True)
                if confidence < 0.5:
                    st.warning(f" Confidence is low ({confidence:.2f}). Double-check facts.")
                st.markdown("#### Source Chunks")
                if sources:
                    for i, source in enumerate(sources):
                        meta = source.get("metadata", {})
                        with st.expander(f"Source {i+1}: {meta.get('source', 'unknown')}"):
                            st.code(source["content"])
                else:
                    st.info("No source chunks found for this answer.")
            else:
                st.error(f"Error: {r.json().get('detail', r.text)}")
        except Exception as ex:
            st.error(f"API call failed: {ex}")
