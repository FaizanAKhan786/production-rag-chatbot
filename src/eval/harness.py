"""Evaluation runner for benchmarking retrieval recall, answer groundedness, and latency."""
import json
import time
from pathlib import Path
from typing import Optional

from src.config import settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.ingestion.embedder import ChunkEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.llm.client import OllamaClient
from src.llm.prompting import assemble_context, build_messages
from src.eval.metrics import recall_at_k, groundedness_keyword, compute_latency_stats
from src.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationHarness:
    """Runs systematic benchmarks over held-out QA pairs."""

    def __init__(
        self,
        qa_pairs_path: Optional[str] = None,
        run_generation: bool = True,
    ):
        self.qa_pairs_path = qa_pairs_path or str(
            Path(__file__).parent / "fixtures" / "qa_pairs.json"
        )
        self.run_generation = run_generation

        # Initialize retrieval components
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        bm25_path = str(settings.bm25_pickle_path)
        if Path(bm25_path).exists():
            self.bm25_index.load(bm25_path)
        
        self.embedder = ChunkEmbedder()
        self.reranker = Reranker()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder,
        )
        self.llm_client = OllamaClient() if run_generation else None

    def run(self) -> dict:
        """Executes evaluation across all test items."""
        with open(self.qa_pairs_path, "r", encoding="utf-8") as f:
            test_items = json.load(f)

        results = []
        retrieval_latencies = []
        rerank_latencies = []
        llm_latencies = []
        recalls_at_1 = []
        recalls_at_3 = []
        recalls_at_5 = []
        keyword_recalls = []
        grounded_ratios = []

        llm_healthy = False
        if self.run_generation and self.llm_client:
            import asyncio
            try:
                llm_healthy = asyncio.run(self.llm_client.health_check())
            except Exception:
                llm_healthy = False

        for item in test_items:
            query = item["question"]
            expected_sources = item.get("expected_sources", [])
            expected_keywords = item.get("expected_keywords", [])

            # 1. Retrieval
            t0 = time.perf_counter()
            candidates = self.hybrid_retriever.retrieve(query)
            t1 = time.perf_counter()
            retrieval_ms = (t1 - t0) * 1000
            retrieval_latencies.append(retrieval_ms)

            # 2. Rerank
            t2 = time.perf_counter()
            scored_chunks = self.reranker.rerank(query, candidates, top_k=5)
            t3 = time.perf_counter()
            rerank_ms = (t3 - t2) * 1000
            rerank_latencies.append(rerank_ms)

            # Extract retrieved source filenames
            retrieved_sources = [
                c.get("metadata", {}).get("source_file", "")
                for c in scored_chunks
            ]

            r1 = recall_at_k(retrieved_sources, expected_sources, k=1)
            r3 = recall_at_k(retrieved_sources, expected_sources, k=3)
            r5 = recall_at_k(retrieved_sources, expected_sources, k=5)

            recalls_at_1.append(r1)
            recalls_at_3.append(r3)
            recalls_at_5.append(r5)

            # 3. LLM Generation (if requested and healthy)
            answer = ""
            llm_ms = 0.0
            if self.run_generation and self.llm_client and llm_healthy:
                import asyncio
                context = assemble_context(scored_chunks)
                messages = build_messages(query, context)
                t4 = time.perf_counter()
                try:
                    answer = asyncio.run(self.llm_client.generate(messages))
                    t5 = time.perf_counter()
                    llm_ms = (t5 - t4) * 1000
                    llm_latencies.append(llm_ms)
                except Exception as e:
                    logger.error("eval_generation_error", error=str(e))
                    answer = f"Error: {e}"

            # 4. Groundedness metrics
            retrieved_context = " ".join([c.get("text", "") for c in scored_chunks])
            groundedness = groundedness_keyword(answer, retrieved_context, expected_keywords)
            keyword_recalls.append(groundedness["keyword_recall"])
            grounded_ratios.append(groundedness["context_grounded_ratio"])

            results.append({
                "id": item.get("id"),
                "question": query,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources[:3],
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "retrieval_ms": round(retrieval_ms, 2),
                "rerank_ms": round(rerank_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "groundedness": groundedness,
            })

        summary = {
            "total_queries": len(test_items),
            "mean_recall@1": round(float(sum(recalls_at_1) / len(recalls_at_1)), 4) if recalls_at_1 else 0.0,
            "mean_recall@3": round(float(sum(recalls_at_3) / len(recalls_at_3)), 4) if recalls_at_3 else 0.0,
            "mean_recall@5": round(float(sum(recalls_at_5) / len(recalls_at_5)), 4) if recalls_at_5 else 0.0,
            "mean_keyword_recall": round(float(sum(keyword_recalls) / len(keyword_recalls)), 4) if keyword_recalls else 0.0,
            "mean_grounded_ratio": round(float(sum(grounded_ratios) / len(grounded_ratios)), 4) if grounded_ratios else 0.0,
            "retrieval_latency": compute_latency_stats(retrieval_latencies),
            "rerank_latency": compute_latency_stats(rerank_latencies),
            "llm_latency": compute_latency_stats(llm_latencies) if llm_latencies else {"status": "skipped_or_offline"},
            "detailed_results": results,
        }

        return summary
