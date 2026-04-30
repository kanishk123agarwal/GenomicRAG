from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_documents(pdf_dir: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """Load all PDFs from a directory and split into overlapping chunks."""
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages → {len(chunks)} chunks")
    return chunks
