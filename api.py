"""
FastAPI backend for RAG system.
- POST /ingest: triggers ingestion
- POST /query: accepts {'question': ...}, returns answer + sources
- GET /health: service status + document count
Full CORS & robust error handling.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ingest import main as ingest_main
from rag_chain import query as rag_query, get_vectorstore

load_dotenv()

app = FastAPI(
    title="RAG Knowledge Base API",
    description="REST API backend for RAG QA system",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/health")
def health():
    try:
        vectordb = get_vectorstore()
        collection = vectordb._collection
        count = collection.count()
        return {"status": "ok", "doc_count": count}
    except Exception as ex:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(ex)})

@app.post("/ingest")
def ingest():
    try:
        ingest_main()
        return {"status": "ok", "message": "Ingestion complete."}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {ex}")

@app.post("/query")
def query(payload: dict):
    question = payload.get("question", "")
    try:
        result = rag_query(question)
        return result
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Query failed: {ex}")

@app.exception_handler(404)
def notfound(req: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})

@app.exception_handler(500)
def server_error(req: Request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
