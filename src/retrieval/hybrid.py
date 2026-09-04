from collections import defaultdict
import time

import numpy as np

from src.config import settings
from src.ingestion.embedder import ChunkEmbedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore
from src.logging_config import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF(d) = Σ 1/(k + rank_m(d)) for each ranking system m.
    
    Args:
        ranked_lists: List of ranked doc ID lists (each sorted by relevance).
        k: Smoothing constant (default 60, per Cormack et al. 2009).
    
    Returns:
        List of (doc_id, rrf_score) tuples, sorted by score descending.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Hybrid retrieval combining BM25 sparse + dense vector search with RRF fusion."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: ChunkEmbedder,
    ) -> None:
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._embedder = embedder
    
    def retrieve(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[dict]:
        """Run hybrid retrieval: BM25 + vector → RRF fusion.
        
        Args:
            query: User query text.
            bm25_top_k: Top-K for BM25 retrieval.
            vector_top_k: Top-K for vector retrieval.
            rrf_k: RRF smoothing constant.
        
        Returns:
            List of candidate dicts sorted by fused RRF score, each with:
            chunk_id, text, metadata, rrf_score.
        """
        bm25_top_k = bm25_top_k or settings.bm25_k_retrieve
        vector_top_k = vector_top_k or settings.vector_k_retrieve
        rrf_k = rrf_k or settings.rrf_k
        
        t_start = time.perf_counter()
        
        # 1. BM25 retrieval
        if self._bm25.is_built and self._bm25.size > 0:
            bm25_results = self._bm25.query(query, top_k=bm25_top_k)
        else:
            bm25_results = []
        bm25_ranked_ids = [cid for cid, _ in bm25_results]
        t_bm25 = time.perf_counter()
        
        # 2. Dense vector retrieval
        query_embedding = self._embedder.embed_query(query)
        vector_results = self._vector_store.query(query_embedding, top_k=vector_top_k)
        vector_ranked_ids = [r["chunk_id"] for r in vector_results]
        t_vector = time.perf_counter()
        
        # 3. RRF Fusion
        fused = reciprocal_rank_fusion(
            [bm25_ranked_ids, vector_ranked_ids],
            k=rrf_k,
        )
        t_rrf = time.perf_counter()
        
        # 4. Fetch full chunk data for fused candidates
        fused_ids = [cid for cid, _ in fused]
        
        # Build lookup from vector results
        chunk_data: dict[str, dict] = {}
        for r in vector_results:
            chunk_data[r["chunk_id"]] = r
        
        # Fetch any BM25-only chunks not already in vector results
        missing_ids = [cid for cid in fused_ids if cid not in chunk_data]
        if missing_ids:
            extra = self._vector_store.get_chunks(missing_ids)
            for r in extra:
                chunk_data[r["chunk_id"]] = r
        
        # Assemble results
        results = []
        for chunk_id, rrf_score in fused:
            if chunk_id in chunk_data:
                data = chunk_data[chunk_id]
                results.append({
                    "chunk_id": chunk_id,
                    "text": data.get("text", ""),
                    "metadata": data.get("metadata", {}),
                    "rrf_score": rrf_score,
                })
        
        logger.info(
            "hybrid_retrieval_complete",
            query_preview=query[:50],
            bm25_hits=len(bm25_results),
            vector_hits=len(vector_results),
            fused_candidates=len(results),
            bm25_ms=round((t_bm25 - t_start) * 1000, 1),
            vector_ms=round((t_vector - t_bm25) * 1000, 1),
            rrf_ms=round((t_rrf - t_vector) * 1000, 1),
        )
        
        return results
