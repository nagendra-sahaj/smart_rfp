from __future__ import annotations
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from src.config import GROQ_API_KEY, GROQ_MODEL_NAME

def build_custom_rag_chain(db, top_k: int, prompt_template: str = None, groq_api_key: str | None = None, groq_model: str | None = None):
    """Build a RAG chain step by step with a custom or default prompt template."""
    api_key = groq_api_key or GROQ_API_KEY
    model = groq_model or GROQ_MODEL_NAME
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    llm = ChatGroq(model=model, api_key=api_key)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # Default prompt template for strict context-based answering
    default_prompt = (
        "You are an expert assistant for contract and RFP analysis.\n"
        "Answer the user's question strictly using only the provided context below.\n"
        "If the context does not contain enough information to answer, reply: 'I am not sure about that, provided context is not sufficient to answer.'\n"
        "\n"
        "Context:\n{context}\n"
        "\n"
        "Question: {question}\n"
        "\n"
        "Answer:"
    )
    prompt_template = prompt_template or default_prompt
    prompt = PromptTemplate.from_template(prompt_template)

    # Chain: retrieve -> format prompt -> LLM -> parse output
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def chain_with_sources(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        answer = llm.invoke(prompt.format(context=context, question=question))
        parsed = StrOutputParser().invoke(answer)
        return {"result": parsed, "source_documents": docs}

    return RunnableLambda(chain_with_sources)


def setup_rag_chain(db, top_k: int, groq_api_key: str | None = None, groq_model: str | None = None):
    """Set up RAG chain with Groq LLM, using config defaults when not provided."""
    api_key = groq_api_key or GROQ_API_KEY
    model = groq_model or GROQ_MODEL_NAME
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    llm = ChatGroq(model=model, api_key=api_key)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    # Custom chain to return both answer and source documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


