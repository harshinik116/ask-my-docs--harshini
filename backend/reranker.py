"""
Cross-encoder reranking using Groq LLM as a relevance judge.

Each candidate chunk is scored independently against the query
(0-10 scale), then results are sorted by score — equivalent in
effect to a cross-encoder that jointly encodes query + passage.
"""
import os
import json
from typing import List, Dict
from groq import Groq


def rerank(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Score each doc against the query using Groq, return top_k sorted by score.
    Falls back to original order if scoring fails.
    """
    if not docs:
        return []

    if len(docs) <= 1:
        return docs[:top_k]

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Build the scoring prompt
    passages = "\n\n".join(
        f'[{i+1}] (Document: {doc["document"]}, Page {doc["page"]})\n{doc["text"][:400]}'
        for i, doc in enumerate(docs)
    )

    prompt = f"""You are a relevance scoring system.

Score each passage's relevance to the query on a scale of 0 to 10.
0 = completely irrelevant, 10 = perfectly answers the query.

Query: "{query}"

Passages:
{passages}

Respond with ONLY valid JSON — a list of objects with "index" (1-based) and "score":
Example: [{{"index": 1, "score": 8}}, {{"index": 2, "score": 3}}]

JSON:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return docs[:top_k]

        scores_data = json.loads(raw[start:end])
        score_map = {item["index"]: item["score"] for item in scores_data}

        # Attach scores and sort
        scored_docs = []
        for i, doc in enumerate(docs):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score_map.get(i + 1, 0)
            scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:top_k]

    except Exception:
        # Graceful fallback: return original order
        return docs[:top_k]
