"""Evaluation metrics for RAG retrieval and answer generation."""
import numpy as np
import re
from typing import Sequence


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int = 5,
) -> float:
    """Calculates recall@K.
    
    Returns 1.0 if at least one expected source is present in top-K retrieved sources,
    otherwise 0.0.
    """
    if not expected_sources:
        return 1.0
    
    top_k_retrieved = set(retrieved_sources[:k])
    expected = set(expected_sources)
    
    matched = top_k_retrieved.intersection(expected)
    return len(matched) / len(expected)


def groundedness_keyword(
    answer: str,
    retrieved_context: str,
    expected_keywords: Sequence[str],
) -> dict[str, float]:
    """Evaluates keyword coverage between generated answer, context, and ground truth keywords.
    
    Checks:
    1. expected_keyword_recall: fraction of expected_keywords present in answer.
    2. context_grounded_fraction: fraction of words in answer also found in retrieved context.
    """
    if not answer.strip():
        return {"keyword_recall": 0.0, "context_grounded_ratio": 0.0}

    # Keyword recall
    matched_keywords = 0
    lower_answer = answer.lower()
    for kw in expected_keywords:
        if kw.lower() in lower_answer:
            matched_keywords += 1
    kw_recall = matched_keywords / len(expected_keywords) if expected_keywords else 1.0

    # Answer words present in context
    answer_words = set(re.findall(r"\w+", lower_answer))
    context_words = set(re.findall(r"\w+", retrieved_context.lower()))

    # Filter short stop words
    informative_answer_words = {w for w in answer_words if len(w) > 3}
    if not informative_answer_words:
        grounded_ratio = 1.0
    else:
        grounded_words = informative_answer_words.intersection(context_words)
        grounded_ratio = len(grounded_words) / len(informative_answer_words)

    return {
        "keyword_recall": round(kw_recall, 4),
        "context_grounded_ratio": round(grounded_ratio, 4),
    }


def compute_latency_stats(durations_ms: Sequence[float]) -> dict[str, float]:
    """Computes p50, p95, and average latency metrics in milliseconds."""
    if not durations_ms:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    
    arr = np.array(durations_ms)
    return {
        "avg_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }
