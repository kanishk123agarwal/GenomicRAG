import os
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def is_genetics_paper(sample_text: str) -> bool:
    """Use Gemini to check if a research paper belongs to the genetics/genomics domain."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return True # Failsafe default
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = (
            "Analyze the following text snippet from the beginning of a research paper.\n"
            "Determine if this paper is related to genetics, genomics, DNA, RNA, CRISPR, gene editing, "
            "hereditary diseases, gene therapy, or molecular biology.\n"
            "Respond with exactly one word: 'Yes' if it is related, or 'No' if it is not.\n\n"
            f"Snippet:\n{sample_text[:1200]}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip().lower()
        return "yes" in result
    except Exception as e:
        print(f"Error validating document domain: {e}")
        return True # Failsafe fallback

def clean_document_text(text: str) -> str:
    """Filter out license notices, headers, footers, and extra spacing line-by-line."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        lower_line = line.lower()
        # Skip IEEE Xplore download watermarks & restrictions
        if "authorized licensed use" in lower_line or "downloaded on" in lower_line or "restrictions apply" in lower_line or "ieee xplore" in lower_line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def load_and_chunk_documents(pdf_dir: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """Load all PDFs from a directory, validate they are genetics papers, clean raw text, and split into chunks."""
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()

    # Group documents by source file to validate each file once
    docs_by_source = defaultdict(list)
    for doc in documents:
        docs_by_source[doc.metadata.get("source")].append(doc)

    valid_documents = []
    rejected_files = []

    for source_path, docs in docs_by_source.items():
        sample_text = docs[0].page_content if docs else ""
        filename = os.path.basename(source_path)
        
        if is_genetics_paper(sample_text):
            for doc in docs:
                doc.page_content = clean_document_text(doc.page_content)
                valid_documents.append(doc)
        else:
            rejected_files.append(filename)
            try:
                if os.path.exists(source_path):
                    os.remove(source_path)
            except Exception:
                pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(valid_documents)
    print(f"Loaded {len(valid_documents)} pages → {len(chunks)} chunks. Rejected: {rejected_files}")
    return chunks, rejected_files
