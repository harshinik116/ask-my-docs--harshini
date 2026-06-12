"""
RAG pipeline with:
  1. Hybrid retrieval  (BM25 + vector search via RRF)
  2. Cross-encoder reranking  (Groq LLM as relevance judge)
  3. Citation-enforced answer generation
"""
import os
from groq import Groq
from backend.hybrid_retrieval import hybrid_search
from backend.reranker import rerank


def answer_question(query: str) -> str:
    # ── Step 1: Hybrid retrieval ──────────────────────────────
    candidates = hybrid_search(query, top_k=10)

    if not candidates:
        return "No documents have been uploaded yet. Please upload a PDF first."

    # ── Step 2: Cross-encoder reranking ──────────────────────
    top_docs = rerank(query, candidates, top_k=5)

    # ── Step 3: Build context with citations ──────────────────
    context = ""
    for i, doc in enumerate(top_docs):
        context += (
            f"Source [{i+1}]\n"
            f"Document: {doc['document']}  |  Page: {doc['page']}\n"
            f"{doc['text']}\n\n"
        )

    # ── Step 4: Generate answer with citation enforcement ─────
    prompt = f"""You are a precise document assistant.

Rules:
1. Answer ONLY using the provided sources below.
2. Every factual claim MUST include a citation in the format [Source N].
3. If the answer cannot be found in the sources, respond with exactly:
   "The answer is not found in the uploaded documents."
4. Never make up information.

Sources:
{context}

Question: {query}

Answer (with citations):"""

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
