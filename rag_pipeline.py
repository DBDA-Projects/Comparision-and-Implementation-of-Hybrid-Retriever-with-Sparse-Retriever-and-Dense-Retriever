from retrieval.hybrid_retriever import HybridRetriever
from llm.gemini_client import generate_answer

retriever = HybridRetriever()

def is_comparative(query):
    keywords = ["compare", "difference", "vs", "versus"]
    return any(k in query.lower() for k in keywords)

def rag_answer(query, k=3):
    comparative = is_comparative(query)

    if comparative:
        docs = retriever.search(query, k=10)
    else:
        docs = retriever.search(query, k)

    context = "\n\n".join(d["text"][:600] for d in docs)
    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "is_comparative": comparative
    }
