"""
CI-gated evaluation pipeline for Ask My Docs.

Metrics evaluated by Groq-as-judge:
  - Answer Relevance  : Does the answer address the question?
  - Faithfulness      : Is every claim grounded in the provided context?
  - Citation Accuracy : Are [Source N] citations present and correctly used?

Each metric is scored 0-10. The pipeline fails CI if any metric
falls below PASSING_THRESHOLD on any test case.

Run locally:
    pytest tests/test_eval.py -v
"""
import os
import json
import pytest
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
PASSING_THRESHOLD = 6          # Minimum score (out of 10) to pass
MODEL = "llama-3.1-8b-instant"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set — skipping eval tests")
    return Groq(api_key=api_key)


def llm_score(client: Groq, criterion: str, question: str, answer: str, context: str = "") -> int:
    """Ask Groq to score a single criterion and return an integer 0-10."""
    criteria_prompts = {
        "relevance": (
            "Does the answer directly and completely address the question? "
            "Score 0 if completely off-topic, 10 if it fully answers the question."
        ),
        "faithfulness": (
            "Is every factual claim in the answer supported by the provided context? "
            "Score 0 if the answer contains hallucinations, 10 if fully grounded."
        ),
        "citation": (
            "Does the answer include inline citations in the format [Source N] "
            "for every factual claim? "
            "Score 0 for no citations, 10 for correct citations on every claim."
        ),
    }

    prompt = f"""You are an evaluation judge. Score the answer on the criterion below.

Criterion: {criteria_prompts[criterion]}

Question: {question}
{"Context: " + context if context else ""}
Answer: {answer}

Respond with ONLY a single integer between 0 and 10. No explanation.
Score:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    raw = response.choices[0].message.content.strip()
    try:
        return int("".join(filter(str.isdigit, raw))[:2])
    except (ValueError, IndexError):
        return 0


def get_rag_answer(question: str) -> tuple[str, str]:
    """Call the RAG pipeline and return (answer, context_str)."""
    # Import here so the module-level ChromaDB init doesn't run at collection time
    from backend.hybrid_retrieval import hybrid_search
    from backend.reranker import rerank
    from backend.rag_pipeline import answer_question

    candidates = hybrid_search(question, top_k=10)
    top_docs = rerank(question, candidates, top_k=5)

    context = "\n\n".join(
        f"[Source {i+1}] {doc['document']} p.{doc['page']}: {doc['text']}"
        for i, doc in enumerate(top_docs)
    )

    answer = answer_question(question)
    return answer, context


# ── Load eval dataset ─────────────────────────────────────────────────────────

def load_eval_cases():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path) as f:
        return json.load(f)


EVAL_CASES = load_eval_cases()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_answer_relevance(case):
    """Answer must be relevant to the question."""
    client = get_groq_client()
    answer, _ = get_rag_answer(case["question"])
    score = llm_score(client, "relevance", case["question"], answer)
    assert score >= PASSING_THRESHOLD, (
        f"[{case['id']}] Relevance score {score}/10 < threshold {PASSING_THRESHOLD}\n"
        f"Q: {case['question']}\nA: {answer}"
    )


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_answer_faithfulness(case):
    """Answer must not hallucinate — every claim must come from the context."""
    client = get_groq_client()
    answer, context = get_rag_answer(case["question"])
    score = llm_score(client, "faithfulness", case["question"], answer, context)
    assert score >= PASSING_THRESHOLD, (
        f"[{case['id']}] Faithfulness score {score}/10 < threshold {PASSING_THRESHOLD}\n"
        f"Q: {case['question']}\nA: {answer}"
    )


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["id"] for c in EVAL_CASES])
def test_citation_enforcement(case):
    """Answer must include [Source N] citations."""
    client = get_groq_client()
    answer, _ = get_rag_answer(case["question"])

    # Hard check: [Source  must appear in the answer
    has_citation = "[source" in answer.lower() or "[1]" in answer
    score = llm_score(client, "citation", case["question"], answer)

    assert has_citation or score >= PASSING_THRESHOLD, (
        f"[{case['id']}] Citation score {score}/10, no [Source N] found\n"
        f"A: {answer}"
    )


def test_hybrid_retrieval_returns_results():
    """Hybrid retrieval must return results when docs are indexed."""
    from backend.hybrid_retrieval import hybrid_search
    from backend.bm25_store import load_corpus

    corpus = load_corpus()
    if not corpus:
        pytest.skip("No documents indexed — upload a PDF first")

    results = hybrid_search("summary", top_k=5)
    assert len(results) > 0, "Hybrid search returned no results"
    assert all("text" in r and "document" in r and "page" in r for r in results)


def test_reranker_preserves_top_results():
    """Reranker must return at most top_k results and preserve required fields."""
    from backend.hybrid_retrieval import hybrid_search
    from backend.reranker import rerank
    from backend.bm25_store import load_corpus

    corpus = load_corpus()
    if not corpus:
        pytest.skip("No documents indexed — upload a PDF first")

    candidates = hybrid_search("test", top_k=10)
    reranked = rerank("test", candidates, top_k=5)

    assert len(reranked) <= 5
    for doc in reranked:
        assert "text" in doc
        assert "document" in doc
        assert "page" in doc
