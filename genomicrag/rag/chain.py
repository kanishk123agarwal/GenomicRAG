from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a genetics research assistant. Use only the provided context to answer the question.
If the provided context is not related to genetics, genomics, DNA/RNA, hereditary diseases, gene therapy, or molecular biology, refuse to answer and say: "I can only answer questions based on genetics research papers."
If the answer is not found in the context, say "I couldn't find this in the uploaded papers."
Always mention which paper or source you are referencing.

Context:
{summaries}

Question: {question}

Answer:"""

def build_chain(vectorstore, model_name: str = "gemini-2.5-flash", search_filter: dict = None):
    """Connect retriever to Google Gemini LLM and return RetrievalQA chain."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
    search_kwargs = {"k": 4}
    if search_filter:
        search_kwargs["filter"] = search_filter
        
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

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
