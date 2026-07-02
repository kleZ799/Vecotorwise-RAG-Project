# RAG Knowledge Base System

A fully-featured Retrieval-Augmented Generation (RAG) system using FastAPI, LangChain, ChromaDB, OpenAI, and Streamlit.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env  # Edit with your OpenAI API key
python ingest.py
uvicorn api:app --reload
streamlit run app.py
```

## 5 Example Test Questions

1. How many vacation days do I get per year?
2. What is AcmeTech’s remote work policy?
3. What is the maximum expense reimbursement allowed per month?
4. How do I claim my equipment allowance?
5. Can I carry over unused vacation days?


## System Overview
- **ingest.py** — Document loading, chunking, embedding, and storage
- **rag_chain.py** — Retrieval and QA logic using GPT-4.1
- **api.py** — FastAPI backend: ingestion, querying, health
- **app.py** — Streamlit frontend UI
- **data/** — Document storage directory
- **.env.example** — Environment/config template


## Platform Support
- Runs on Windows, Mac, and Linux (Python 3.10+ required)
- No external vector DB needed (ChromaDB stores locally)
---

## Troubleshooting
- Ensure `.env` has your OpenAI API key set
- Run `python ingest.py` after adding or changing documents
- If errors occur, check logs and confirm all dependencies installed with correct versions
