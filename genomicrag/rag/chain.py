from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a genetics research assistant. Use only the provided context to answer the question.
If the answer is not found in the context, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{summaries}

Question: {question}

Answer:"""

def build_chain(vectorstore, model_name: str = "gemini-1.5-flash"):
    """
    [Skeleton] Connect retriever to Google Gemini LLM and return RetrievalQA chain.
    Will be fully implemented in Phase 4.
    """
    print(f"[Skeleton] Building RAG chain with model {model_name}...")
    # Return None as placeholder for Phase 1
    return None
