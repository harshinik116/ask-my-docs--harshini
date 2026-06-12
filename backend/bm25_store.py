import json
import os
from rank_bm25 import BM25Okapi
from typing import List, Dict

CORPUS_PATH = "vector_store/bm25_corpus.json"


def load_corpus() -> List[Dict]:
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_corpus(corpus: List[Dict]):
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)


def add_to_corpus(chunks: List[Dict], document_name: str):
    corpus = load_corpus()
    existing_ids = {doc["id"] for doc in corpus}
    for i, chunk in enumerate(chunks):
        doc_id = f"{document_name}_{i}"
        if doc_id not in existing_ids:
            corpus.append({
                "id": doc_id,
                "text": chunk["text"],
                "document": document_name,
                "page": chunk["page"]
            })
    save_corpus(corpus)


def bm25_search(query: str, top_k: int = 10) -> List[Dict]:
    corpus = load_corpus()
    if not corpus:
        return []

    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": corpus[i]["id"],
            "text": corpus[i]["text"],
            "document": corpus[i]["document"],
            "page": corpus[i]["page"],
            "bm25_score": float(scores[i])
        }
        for i in top_indices
        if scores[i] > 0
    ]
