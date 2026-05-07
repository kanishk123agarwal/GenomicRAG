import streamlit as st
import os
import glob
import shutil
import re
from dotenv import load_dotenv
from rag.loader import load_and_chunk_documents
from rag.embedder import build_vectorstore, load_vectorstore
from rag.chain import build_chain

# Load environment variables from .env file relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"), override=True)

st.set_page_config(page_title="GenomicRAG", page_icon="🔬", layout="wide")

# Inject premium custom styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Hide default Streamlit Deploy button */
    .stDeployButton {
        display: none !important;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #6366f1, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .subtitle-caption {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 15px;
    }
    .stat-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .stat-icon {
        font-size: 1.8rem;
        background: rgba(99, 102, 241, 0.15);
        border-radius: 8px;
        padding: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6366f1;
    }
    .stat-number {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color);
        line-height: 1;
    }
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .answer-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6366f1;
        margin-top: 25px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .source-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .answer-box {
        background-color: rgba(99, 102, 241, 0.07);
        border-left: 4px solid #6366f1;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 10px;
        margin-bottom: 20px;
        line-height: 1.7;
        font-size: 1.1rem;
        color: var(--text-color);
    }
    .source-box {
        background-color: transparent;
        color: var(--text-color);
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

# Render main header
st.markdown('<h1 class="main-title">🔬 GenomicRAG</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-caption">Ask questions about your genetics research papers — Powered by Google Gemini</p>', unsafe_allow_html=True)

# Load existing vectorstore and initialize the RAG chain if available
if "chain" not in st.session_state:
    if os.path.exists("vectorstore/faiss_index"):
        vs = load_vectorstore()
        if vs is not None:
            st.session_state.chain = build_chain(vs)
            st.session_state.vectorstore = vs
        else:
            st.session_state.chain = None
            st.session_state.vectorstore = None
    else:
        st.session_state.chain = None
        st.session_state.vectorstore = None

# Sidebar for uploading and indexing papers
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔬 Settings & Upload</div>', unsafe_allow_html=True)
    
    # Check if Gemini API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        st.warning("⚠️ **GOOGLE_API_KEY is not configured.** Please add your Gemini API key to your `.env` file.")

    st.header("Upload Papers")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if st.button("Index Papers", use_container_width=True) and uploaded_files:
        os.makedirs("data/papers", exist_ok=True)
        for f in uploaded_files:
            with open(f"data/papers/{f.name}", "wb") as out:
                out.write(f.read())
        with st.spinner("Loading, validating, and chunking PDFs..."):
            chunks, rejected = load_and_chunk_documents("data/papers")
            
        if chunks:
            with st.spinner("Embedding chunks and building FAISS index..."):
                vs = build_vectorstore(chunks)
                if vs is not None:
                    st.session_state.chain = build_chain(vs)
                    st.session_state.vectorstore = vs
        
        # Render warnings for rejected papers
        if rejected:
            st.warning(f"⚠️ Rejected non-genetics papers: {', '.join(rejected)}")
            
        # Render success for valid papers
        indexed_count = len(uploaded_files) - len(rejected)
        if indexed_count > 0 and chunks:
            st.success(f"Successfully indexed {indexed_count} genetics paper(s)!")
        elif not chunks:
            st.error("No valid genetics papers were indexed. Please upload genomics/genetics PDFs.")

    # Calculate statistics
    papers_count = len(glob.glob("data/papers/*.pdf"))
    chunks_count = 0
    if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
        try:
            chunks_count = st.session_state.vectorstore.index.ntotal
        except Exception:
            pass

    # Display statistics
    st.markdown("---")
    st.markdown("### Library Statistics")
    
    # Render papers stat card
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">📄</div>
            <div>
                <div class="stat-number">{papers_count}</div>
                <div class="stat-label">Papers Uploaded</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Render chunks stat card
    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">🧩</div>
            <div>
                <div class="stat-number">{chunks_count}</div>
                <div class="stat-label">Text Chunks</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Clear Index Action
    if papers_count > 0 or chunks_count > 0:
        st.markdown("---")
        if st.button("🗑️ Clear Index & Library", use_container_width=True):
            if os.path.exists("data/papers"):
                shutil.rmtree("data/papers")
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            st.session_state.chain = None
            st.session_state.vectorstore = None
            st.success("Successfully cleared local index!")
            st.rerun()

# Build the layout for the query space
paper_files = glob.glob("data/papers/*.pdf")
paper_names = [os.path.basename(p) for p in paper_files]

col1, col2 = st.columns([3, 1])
with col2:
    query_scope = st.selectbox("Query Scope", ["All Papers"] + paper_names)
with col1:
    question = st.text_input("Ask a question about your papers")

def format_source_content(content: str) -> str:
    """Format and clean source content to look cohesive and reflow beautifully in the UI."""
    content = re.sub(r' +', ' ', content)
    # Reflow single newlines into spaces to clean page-breaks, keeping double newlines for paragraph boundaries
    paragraphs = content.split('\n\n')
    cleaned_paragraphs = []
    for p in paragraphs:
        cleaned_p = p.replace('\n', ' ').strip()
        # Clean up double spacing created by replacements
        cleaned_p = re.sub(r'\s+', ' ', cleaned_p)
        cleaned_paragraphs.append(cleaned_p)
    return '\n\n'.join(cleaned_paragraphs)

if question:
    if "chain" in st.session_state and st.session_state.chain is not None:
        with st.spinner("Searching papers & synthesizing answer..."):
            try:
                # Dynamically compile the correct chain based on selected query scope filter
                if query_scope != "All Papers" and st.session_state.vectorstore is not None:
                    # Resolve path structure mapping in vector store (relative vs absolute)
                    try:
                        sample_doc = list(st.session_state.vectorstore.docstore._dict.values())[0]
                        sample_source = sample_doc.metadata.get("source", "")
                        if os.path.isabs(sample_source):
                            source_filter_path = os.path.abspath(os.path.join("data/papers", query_scope))
                        else:
                            source_filter_path = os.path.join("data/papers", query_scope)
                    except Exception:
                        source_filter_path = os.path.join("data/papers", query_scope)

                    # Build filtered RAG chain
                    chain_to_use = build_chain(
                        st.session_state.vectorstore,
                        search_filter={"source": source_filter_path}
                    )
                else:
                    chain_to_use = st.session_state.chain

                # Run query
                result = chain_to_use({"question": question})
                
                # Render Answer Header and Custom Box
                st.markdown('<div class="answer-header">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">{result.get("answer", "No answer generated.")}</div>', unsafe_allow_html=True)
                
                # Render Sources Header
                st.markdown('<div class="source-header">Retrieved Context Sources</div>', unsafe_allow_html=True)
                source_docs = result.get("source_documents", [])
                if source_docs:
                    # Group source documents by source_name and page_num to deduplicate expanders
                    grouped_sources = {}
                    for doc in source_docs:
                        source_path = doc.metadata.get('source', 'Unknown')
                        source_name = os.path.basename(source_path)
                        page_num = doc.metadata.get('page', 0) + 1
                        
                        key = (source_name, page_num)
                        if key not in grouped_sources:
                            grouped_sources[key] = []
                        
                        cleaned_content = format_source_content(doc.page_content)
                        if cleaned_content not in grouped_sources[key]:
                            grouped_sources[key].append(cleaned_content)
                    
                    for (source_name, page_num), contents in grouped_sources.items():
                        joined_content = "\n\n<hr style='border: 1px dashed rgba(128,128,128,0.25); margin: 15px 0;' />\n\n".join(contents)
                        with st.expander(f"📄 {source_name} — Page {page_num}"):
                            st.markdown(f'<div class="source-box">{joined_content}</div>', unsafe_allow_html=True)
                else:
                    st.info("No sources cited for this response.")
            except Exception as e:
                st.error(f"Error querying Gemini model: {e}")
                st.info("Please make sure you have configured a valid GOOGLE_API_KEY in your .env file.")
    else:
        st.info("💬 Please upload and index papers in the sidebar first to enable search and Q&A!")
