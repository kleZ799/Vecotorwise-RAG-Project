"""
Handles retrieval-augmented generation for QA.
Retrieves top 5 relevant chunks from ChromaDB and queries GPT-4.1 for answers.
Returns answer, source chunks, and confidence score.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_vectorstore():
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError("ChromaDB not found (did you run ingest.py?)")
    embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    vectordb = Chroma(
        collection_name="rag_docs",
        embedding_function=embed,
        persist_directory=CHROMA_DB_PATH
    )
    return vectordb

def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 5, "fetch_k": 8})

def query(question: str) -> Dict[str, Any]:
    """
    Retrieves sources and returns answer and confidence score.
    """
    if not question.strip():
        return {
            "answer": "Input question is empty.",
            "sources": [],
            "confidence": 0.0
        }
    retriever = get_retriever()
    retrieved = retriever.get_relevant_documents(question)
    if not retrieved:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": [],
            "confidence": 0.0
        }
    # Score uses average 'distance' from Chroma; lower means more confident
    dists = [doc.metadata.get("distance", 1.0) for doc in retrieved if "distance" in doc.metadata]
    confidence = 1.0 - min(sum(dists) / max(len(dists),1), 1.0) if dists else 0.8
    llm = ChatOpenAI(
        temperature=0.0,
        model="gpt-4-1106-preview",
        openai_api_key=OPENAI_API_KEY,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )
    try:
        result = qa({"query": question})
        answer = result["result"]
        sources: List[Document] = result["source_documents"]
        return {
            "answer": answer.strip(),
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in sources
            ],
            "confidence": round(confidence, 2)
        }
    except Exception as ex:
        return {
            "answer": f"An error occurred: {ex}",
            "sources": [],
            "confidence": 0.0
        }
