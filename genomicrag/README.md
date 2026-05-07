# GenomicRAG — Genomic Research Assistant

GenomicRAG is a domain-specific Retrieval-Augmented Generation (RAG) web application designed to allow researchers and genetics students to upload academic research papers (PDFs) and ask natural language questions. The application parses the files, verifies they are related to genetics/genomics, indexes their core text (excluding the bibliography lists), and generates answers grounded in the papers' content along with precise citations.

---

## Application Preview

### 1. Main Interface Overview
![Main Interface](images/FrontPage.png)

### 2. PDF Upload & Text Chunking
![PDF Upload](images/Upload_and_chunking.png)

### 3. Factual Q&A with Sources
![Q&A with Sources](images/answer.png)

### 4. Non-Genetics Paper Rejection (Domain Guardrail)
![Domain Validation Guardrail](images/Genetic_papers_only.png)

---

## How It Works

GenomicRAG operates through a multi-stage pipeline:

1. **Domain Verification**: Whenever a PDF is uploaded, a sample snippet is analyzed by Google's Gemini LLM to ensure the document is related to genetics, genomics, molecular biology, or hereditary diseases. Non-relevant documents are automatically rejected to preserve the database's integrity.

2. **Text Cleaning & Reference Trimming**: 
   - Raw text is cleaned of IEEE or publisher headers, footers, and license watermarks.
   - A sequence parsing filter detects the start of the **References** or **Bibliography** section in the PDF and truncates the remaining pages. This prevents bibliographical references from clogging the vector store and polluting retrieved sources.

3. **Chunking & Vector Embeddings**: The text is split into overlapping chunks of 2000 characters using LangChain's `RecursiveCharacterTextSplitter`. These chunks are converted into dense vector embeddings using the `all-MiniLM-L6-v2` Sentence-Transformers model.

4. **FAISS Local Vector Store**: The embeddings are saved into a local FAISS vector database. Since it runs 100% locally on the filesystem, it has zero cost, zero external API latency, and does not require provisioning a database cloud service.

5. **Query Expansion & Hybrid RAG Retrieval**: When a query is entered, a custom `GenomicRetriever` translates general terms or abbreviations (like "BRCA") into specific synonyms and gene symbols (like "BRCA BRCA1 BRCA2 breast cancer gene") to retrieve the most semantically relevant text chunks.

6. **Answer Synthesis**: The retrieved chunks are formatted and sent to the Google Gemini model (`gemini-2.5-flash`) along with the user's question, generating an answer only if supported by the context, accompanied by collapsible citation cards.

---

## Tech Stack & Core Libraries

- **Streamlit**
  Streamlit was chosen because it allows rapid prototyping of machine learning UIs without needing complex frontend development in HTML/JS. In this project, it renders a sleek, interactive sidebar for PDF uploads, displays real-time database library statistics, provides a dropdown for narrowing search queries to specific files, and displays output responses and source citations within clean collapsible containers.

- **LangChain**
  LangChain serves as the main orchestration wrapper that chains together our document load, retrieval, and synthesis phases. It manages the prompt templates, coordinates the document splitting, and structures the overall retrieval flow through `RetrievalQAWithSourcesChain` to ensure that raw context blocks are correctly retrieved and combined before being passed to the Gemini model.

- **FAISS (CPU)**
  Facebook AI Similarity Search (FAISS) is utilized as our local vector database to store and query the generated embeddings. It is extremely fast, works 100% locally on the CPU (negating the need for expensive GPU instances or cloud database setups), and allows saving and loading the index directly from file directories on the disk, making it highly portable.

- **HuggingFace Embeddings**
  We leverage the pre-trained `all-MiniLM-L6-v2` model from Sentence-Transformers to map document chunks into 384-dimensional dense vectors. This model offers a perfect balance between speed, computational lightweightness, and semantic capture accuracy, allowing us to find text snippets that match the conceptual meaning of user queries instead of relying on exact word matches.

- **Google GenAI API (Gemini)**
  The project utilizes Google's `gemini-2.5-flash` model for both domain verification (filtering out non-genetics papers) and final answer synthesis. Gemini was chosen due to its high speed, low latency, advanced reasoning capabilities, and ability to handle complex medical and scientific terminology with absolute factual accuracy.

- **PyPDF**
  PyPDF is employed in the background to handle the initial parsing and loading of raw PDF documents page by page. It parses the document layout, extracts text layers accurately, and preserves page-number metadata, which allows our citation system to map each retrieved chunk back to the exact page of the original document.

- **Python-dotenv**
  Python-dotenv is used to manage local configuration and environment secrets securely. It automatically reads key-value pairs from a local `.env` file at application startup and injects them as system environment variables, ensuring that sensitive API credentials (like the Google Gemini key) are never hardcoded into the source files.

---

## How to Set Up & Run Locally

1. **Navigate to the application folder**:
   ```bash
   cd genomicrag
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Provide your API Key**:
   Create a `.env` file containing your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```
