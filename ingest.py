"""
Ingests all .txt and .pdf files from DATA_PATH, splits into 500-token chunks,
embeds with OpenAI (ada-002), and stores in ChromaDB at CHROMA_DB_PATH.
Idempotent (no duplicated data on re-run). Prints stepwise progress.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "./data")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents(path):
    """Load all .txt and .pdf files as LangChain Documents."""
    txt_loader = DirectoryLoader(path, glob="*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = txt_loader.load() + pdf_loader.load()
    return docs

def split_documents(documents):
    """Split into 500-token chunks with 50-token overlap."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def main():
    print(f"[1/4] Loading all documents from: {DATA_PATH}")
    docs = load_documents(DATA_PATH)
    print(f"Loaded {len(docs)} document(s).")

    print("[2/4] Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunk(s).")

    print(f"[3/4] Initializing local ChromaDB at: {CHROMA_DB_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("rag_docs")

    old_ids = set(collection.get(include=["ids"])["ids"] or [])
    new_texts, new_metadatas, new_ids = [], [], []
    for i, chunk in enumerate(chunks):
        # Unique id for idempotency
        src = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk_id = f"{src}_{page}_{i}"
        if chunk_id in old_ids:
            continue
        new_texts.append(chunk.page_content)
        new_metadatas.append(chunk.metadata)
        new_ids.append(chunk_id)

    print(f"[4/4] New chunks to embed/store: {len(new_texts)}")
    if not new_texts:
        print("No new data to ingest. ChromaDB is up-to-date.")
        return

    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not set in .env")
        return

    embedder = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
    )
    try:
        embeddings = embedder.embed_documents(new_texts)
    except Exception as ex:
        print("❌ Error while embedding:", ex)
        return

    collection.add(
        embeddings=embeddings,
        documents=new_texts,
        metadatas=new_metadatas,
        ids=new_ids
    )
    print(f"✅ Ingested {len(new_texts)} new chunk(s) into ChromaDB.")

if __name__ == "__main__":
    main()
