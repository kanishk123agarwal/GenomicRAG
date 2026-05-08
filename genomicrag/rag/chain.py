from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
import re
import os

PROMPT_TEMPLATE = """You are a genetics research assistant. Use the provided context to answer the question.
Provide a detailed, comprehensive, and well-structured explanation. DO NOT summarize or shorten the explanation. Fully explain all concepts mentioned in the query (for example, if both TP53 and CRISPR are asked about, explain both thoroughly). Draw as much detail from the context for each concept as possible.
If the user asks about a general gene name (like BRCA) or related concepts, relate it to the specific genes (like BRCA1, BRCA2) discussed in the context.
If the context does not explicitly define a gene or term mentioned but discusses it, you may use your general genetic knowledge to briefly define the term, keeping it grounded in the topics present in the context.
If the provided context is not related to genetics, genomics, DNA/RNA, hereditary diseases, gene therapy, or molecular biology, refuse to answer and say: "I can only answer questions based on genetics research papers."
If the answer is not found in the context and cannot be inferred, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_STANDARD = PROMPT_TEMPLATE

PROMPT_TEMPLATE_LARGE = """You are a genetics research assistant. Use the provided context to answer the question.
Provide an extremely detailed, exhaustive, and large-format academic explanation. Break your answer down into logical, clear sections (e.g., using bold markdown headings, bullet points, and paragraphs). For each concept in the query, describe its molecular mechanisms, experimental results, and biological implications in full depth, utilizing every relevant detail from the context. Do not omit any findings, data points, or observations.
If the user asks about a general gene name (like BRCA) or related concepts, relate it to the specific genes (like BRCA1, BRCA2) discussed in the context.
If the context does not explicitly define a gene or term mentioned but discusses it, you may use your general genetic knowledge to define the term extensively, keeping it grounded in the topics present in the context.
If the provided context is not related to genetics, genomics, DNA/RNA, hereditary diseases, gene therapy, or molecular biology, refuse to answer and say: "I can only answer questions based on genetics research papers."
If the answer is not found in the context and cannot be inferred, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_CONCISE = """You are a genetics research assistant. Use the provided context to answer the question.
Provide a concise, direct, and focused summary answering the question. Get straight to the point and summarize the key findings from the context in a brief paragraph or bullet points. Avoid unnecessary elaborations.
If the user asks about a general gene name (like BRCA) or related concepts, relate it to the specific genes (like BRCA1, BRCA2) discussed in the context.
If the provided context is not related to genetics, genomics, DNA/RNA, hereditary diseases, gene therapy, or molecular biology, refuse to answer and say: "I can only answer questions based on genetics research papers."
If the answer is not found in the context and cannot be inferred, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{context}

Question: {question}

Answer:"""

class GenomicRetriever(BaseRetriever):
    base_retriever: Any
    llm: Any
    max_chunks: int = 8

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # 1. Ask Gemini to split the query into sub-queries if it contains multiple topics/questions
        prompt = (
            "You are a search assistant. Split the following user query into 1 to 3 distinct search terms or simpler sub-queries to maximize retrieval recall from a vector database.\n"
            "Each search query must be a simple keyword or phrase.\n"
            "Respond with exactly one search query per line. Do not write numbers, bullet points, introduction, or markdown.\n"
            "If the query is already simple and contains only one topic, just return it as-is.\n\n"
            f"Query: {query}"
        )
        
        try:
            response = self.llm.invoke(prompt)
            content = response if isinstance(response, str) else response.content
            sub_queries = [line.strip() for line in content.strip().split("\n") if line.strip()]
        except Exception:
            sub_queries = [query]

        # 2. Retrieve documents for each sub-query
        all_docs = []
        seen_contents = set()
        for sub_q in sub_queries:
            # Query expansion for BRCA specifically (a common abbreviation)
            expanded_sub_q = sub_q
            if re.search(r'\bbrca\b', sub_q, re.IGNORECASE):
                expanded_sub_q = sub_q + " BRCA1 BRCA2 breast cancer gene"
                
            docs = self.base_retriever.invoke(expanded_sub_q)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    all_docs.append(doc)
                    
        return all_docs[:self.max_chunks]  # Limit to top combined chunks

def build_chain(vectorstore, model_name: str = "gemini-flash-latest", search_filter: dict = None, k: int = 4, max_chunks: int = 8, detail_level: str = "Standard Detail"):
    """Connect retriever to Google Gemini LLM and return RetrievalQA chain."""
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=api_key)
    
    search_kwargs = {"k": k}
    if search_filter:
        search_kwargs["filter"] = search_filter
        
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    retriever = GenomicRetriever(base_retriever=base_retriever, llm=llm, max_chunks=max_chunks)

    # Select prompt template based on detail level
    if "large" in detail_level.lower() or "high" in detail_level.lower():
        prompt_text = PROMPT_TEMPLATE_LARGE
    elif "concise" in detail_level.lower():
        prompt_text = PROMPT_TEMPLATE_CONCISE
    else:
        prompt_text = PROMPT_TEMPLATE_STANDARD

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        input_key="question",
        output_key="answer"
    )
    return chain
