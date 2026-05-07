from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Any
import re
import os

PROMPT_TEMPLATE = """You are a genetics research assistant. Use the provided context to answer the question.
Provide a detailed, comprehensive, and well-structured explanation. Explain the background, molecular mechanisms, or clinical significance of the genetic concepts when relevant, using details from the provided context.
If the user asks about a general gene name (like BRCA) or related concepts, relate it to the specific genes (like BRCA1, BRCA2) discussed in the context.
If the context does not explicitly define a gene or term mentioned but discusses it, you may use your general genetic knowledge to briefly define the term, keeping it grounded in the topics present in the context.
If the provided context is not related to genetics, genomics, DNA/RNA, hereditary diseases, gene therapy, or molecular biology, refuse to answer and say: "I can only answer questions based on genetics research papers."
If the answer is not found in the context and cannot be inferred, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{summaries}

Question: {question}

Answer:"""

class GenomicRetriever(BaseRetriever):
    base_retriever: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Expand common query terms to improve recall (e.g., BRCA -> BRCA1, BRCA2)
        expanded_query = query
        if re.search(r'\bbrca\b', query, re.IGNORECASE):
            expanded_query = query + " BRCA1 BRCA2 breast cancer gene"
        return self.base_retriever.invoke(expanded_query)

def build_chain(vectorstore, model_name: str = "gemini-2.5-flash", search_filter: dict = None):
    """Connect retriever to Google Gemini LLM and return RetrievalQA chain."""
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=api_key)
    
    search_kwargs = {"k": 6}
    if search_filter:
        search_kwargs["filter"] = search_filter
        
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    retriever = GenomicRetriever(base_retriever=base_retriever)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["summaries", "question"]
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain
