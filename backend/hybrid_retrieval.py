"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF)
to combine vector search and BM25 keyword search results.
"""
from typing import List, Dict
from backend.retrieval import search_documents
from backend.bm25_store import bm25_search


def reciprocal_rank_fusion(rankings: List[List[Dict]], k: int = 60) -> Dict[str, float]:
    """Combine multiple ranked lists into a single score using RRF."""
    rrf_scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return rrf_scores


def hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Hybrid search: BM25 + vector search fused with RRF.
    Returns top_k deduplicated and re-ranked results.
    """

    # --- Vector search ---
    vector_raw = search_documents(query, top_k=top_k)
    vector_docs = []
    docs_list = vector_raw.get("documents", [[]])[0]
    metas_list = vector_raw.get("metadatas", [[]])[0]
    ids_list = vector_raw.get("ids", [[]])[0] if vector_raw.get("ids") else []

    for i, (text, meta) in enumerate(zip(docs_list, metas_list)):
        doc_id = ids_list[i] if ids_list else f"{meta['document']}_{i}"
        vector_docs.append({
            "id": doc_id,
            "text": text,
            "document": meta["document"],
            "page": meta["page"]
        })

    # --- BM25 search ---
    bm25_docs = bm25_search(query, top_k=top_k)

    if not vector_docs and not bm25_docs:
        return []

    # --- RRF fusion ---
    rrf_scores = reciprocal_rank_fusion([vector_docs, bm25_docs])

    # Merge into a lookup by ID (prefer vector doc metadata)
    all_docs: Dict[str, Dict] = {}
    for doc in bm25_docs + vector_docs:          # vector overwrites BM25 for same ID
        all_docs[doc["id"]] = doc

    # Sort by RRF score descending
    ranked_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for doc_id in ranked_ids[:top_k]:
        if doc_id in all_docs:
            doc = all_docs[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            results.append(doc)

    return results
