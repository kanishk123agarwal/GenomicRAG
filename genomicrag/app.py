import streamlit as st
import os
from dotenv import load_dotenv
from rag.loader import load_and_chunk_documents
from rag.embedder import build_vectorstore, load_vectorstore
from rag.chain import build_chain

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="GenomicRAG (Phase 3)", page_icon="🔬", layout="wide")
st.title("🔬 GenomicRAG")
st.caption("Ask questions about your genetics research papers — Phase 3: Embedding & FAISS Index")

# Sidebar for uploading and indexing papers
with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if st.button("Index Papers") and uploaded_files:
        os.makedirs("data/papers", exist_ok=True)
        for f in uploaded_files:
            with open(f"data/papers/{f.name}", "wb") as out:
                out.write(f.read())
        with st.spinner("Loading and chunking PDFs..."):
            chunks = load_and_chunk_documents("data/papers")
        with st.spinner("Embedding chunks and building FAISS index..."):
            build_vectorstore(chunks)
        st.success(f"Successfully loaded {len(uploaded_files)} paper(s), split into {len(chunks)} chunks, and built the FAISS vector index!")

# Check if Gemini API key is set
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("⚠️ GOOGLE_API_KEY not found in environment. Please add it to your `.env` file to prepare for future phases.")

# Initialize the RAG chain
if "chain" not in st.session_state:
    if os.path.exists("vectorstore/faiss_index"):
        vs = load_vectorstore()
        if vs is not None:
            st.session_state.chain = build_chain(vs)
        else:
            st.session_state.chain = None
    else:
        st.session_state.chain = None

# Question input area
question = st.text_input("Ask a question about your papers")

if question:
    if "chain" in st.session_state and st.session_state.chain is not None:
        with st.spinner("Searching papers..."):
            # Mock results for Phase 1 if chain exists
            result = st.session_state.chain({"question": question})
        st.markdown("### Answer")
        st.write(result.get("answer", "No answer generated."))
        st.markdown("### Sources")
        for doc in result.get("source_documents", []):
            with st.expander(f"📄 {doc.metadata.get('source', 'Unknown')} — page {doc.metadata.get('page', '?')}"):
                st.write(doc.page_content)
    else:
        st.info("💬 [Phase 3] You asked: *\"" + question + "\"*. Once we implement Phase 4/5 (RAG Chain & UI Integration), you will get real answers from Gemini!")
