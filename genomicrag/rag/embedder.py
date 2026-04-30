import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore/faiss_index"

def build_vectorstore(chunks):
    """
    [Skeleton] Embed chunks and save FAISS index to disk.
    Will be fully implemented in Phase 3.
    """
    print(f"[Skeleton] Building vectorstore from {len(chunks)} chunks...")
    # Return None as placeholder for Phase 1
    return None

def load_vectorstore():
    """
    [Skeleton] Load existing FAISS index from disk.
    Will be fully implemented in Phase 3.
    """
    print(f"[Skeleton] Loading vectorstore from {VECTORSTORE_PATH}...")
    # Return None as placeholder for Phase 1
    return None
