import os
import pickle
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

VECTORSTORE_PATH = "vectorstore/store.pkl"

def get_embeddings():
    """Return Google Generative AI embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


class SimpleVectorStore:
    """Pure-numpy cosine-similarity vector store — no C extensions required."""

    def __init__(self, documents, matrix, embeddings_model):
        self.documents = documents
        self.matrix = matrix          # shape: (N, D), L2-normalised
        self.embeddings_model = embeddings_model

    # ------------------------------------------------------------------ #
    #  Retrieval
    # ------------------------------------------------------------------ #
    def similarity_search(self, query: str, k: int = 6, filter: dict = None):
        q_vec = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 1e-10:
            q_vec /= norm

        scores = self.matrix @ q_vec   # cosine similarity for each doc

        if filter:
            indices = [
                i for i, doc in enumerate(self.documents)
                if all(doc.metadata.get(k) == v for k, v in filter.items())
            ]
            if indices:
                ranked = sorted(indices, key=lambda i: scores[i], reverse=True)
                return [self.documents[i] for i in ranked[:k]]

        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx]

    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 6)
        filter_dict = search_kwargs.get("filter", None)
        return _SimpleRetriever(self, k, filter_dict)


class _SimpleRetriever:
    """Minimal LangChain-compatible retriever wrapping SimpleVectorStore."""

    def __init__(self, store: SimpleVectorStore, k: int, filter_dict: dict):
        self.store = store
        self.k = k
        self.filter_dict = filter_dict

    def invoke(self, query: str):
        return self.store.similarity_search(query, k=self.k, filter=self.filter_dict)

    # Legacy LangChain compatibility
    def get_relevant_documents(self, query: str):
        return self.invoke(query)


# ------------------------------------------------------------------ #
#  Build / Save / Load
# ------------------------------------------------------------------ #
def build_vectorstore(chunks):
    """Embed chunks, build numpy matrix, and pickle to disk."""
    embeddings_model = get_embeddings()
    texts = [doc.page_content for doc in chunks]

    print(f"Embedding {len(texts)} chunks with Google Gemini...")
    vecs = embeddings_model.embed_documents(texts)
    matrix = np.array(vecs, dtype=np.float32)

    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-10)

    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump({"documents": chunks, "matrix": matrix}, f)

    print(f"Vector store saved to {VECTORSTORE_PATH}")
    return SimpleVectorStore(chunks, matrix, embeddings_model)


def load_vectorstore():
    """Load pickled vector store from disk."""
    with open(VECTORSTORE_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings_model = get_embeddings()
    return SimpleVectorStore(data["documents"], data["matrix"], embeddings_model)
