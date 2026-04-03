import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore/faiss_index"

def build_vectorstore(chunks):
    """Embed chunks and save FAISS index to disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # Ensure the parent directories exist
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"FAISS index saved to {VECTORSTORE_PATH}")
    return vectorstore

def load_vectorstore():
    """Load existing FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
