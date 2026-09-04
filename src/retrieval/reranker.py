import time
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3."""
    
    def __init__(self) -> None:
        logger.info("loading_reranker", model=settings.reranker_model, device=settings.reranker_device)
        try:
            self._model = CrossEncoder(
                settings.reranker_model,
                activation_fn=torch.nn.Sigmoid(),
                device=settings.reranker_device,
            )
            self._available = True
            logger.info("reranker_loaded")
        except Exception as e:
            logger.error("reranker_load_failed", error=str(e))
            self._model = None
            self._available = False
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank candidates using the cross-encoder.
        
        Args:
            query: User query.
            candidates: List of dicts with at least 'chunk_id' and 'text'.
            top_k: Number of top results after reranking.
        
        Returns:
            List of candidate dicts with added 'rerank_score', sorted by score descending.
            If all scores below threshold, returns empty list.
            If reranker unavailable, returns candidates sorted by existing rrf_score.
        """
        top_k = top_k or settings.rerank_top_k
        
        if not self._available or not candidates:
            logger.warning("reranker_fallback", reason="unavailable" if not self._available else "no_candidates")
            return candidates[:top_k]
        
        t_start = time.perf_counter()
        
        # Extract texts for reranking
        documents = [c["text"] for c in candidates]
        
        # Use CrossEncoder.rank() for convenience
        try:
            ranked = self._model.rank(
                query=query,
                documents=documents,
                top_k=min(top_k, len(documents)),
                return_documents=False,
            )
        except Exception as e:
            logger.error("reranker_inference_failed", error=str(e))
            return candidates[:top_k]
        
        t_end = time.perf_counter()
        
        # Map back to candidates with rerank scores
        results = []
        for item in ranked:
            idx = item["corpus_id"]
            score = float(item["score"])
            
            if score < settings.similarity_threshold:
                continue
            
            candidate = dict(candidates[idx])  # copy
            candidate["rerank_score"] = score
            results.append(candidate)
        
        logger.info(
            "reranking_complete",
            input_candidates=len(candidates),
            output_candidates=len(results),
            rerank_ms=round((t_end - t_start) * 1000, 1),
            threshold=settings.similarity_threshold,
        )
        
        return results
    
    def health_check(self) -> bool:
        """Check if the reranker is available."""
        return self._available
