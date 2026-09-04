"""Standalone script to test the retrieval pipeline.

Usage: python scripts/test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging
from src.config import settings
from src.ingestion.embedder import ChunkEmbedder
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


def main() -> None:
    setup_logging()
    
    print("="*60)
    print("RAG Chatbot — Retrieval Pipeline Test")
    print("="*60)
    
    # Initialize components
    print("\n[1] Loading vector store...")
    vector_store = VectorStore()
    print(f"    Collection has {vector_store.collection.count()} chunks")
    
    print("\n[2] Loading BM25 index...")
    bm25 = BM25Index()
    bm25_path = str(settings.bm25_pickle_path)
    if Path(bm25_path).exists():
        bm25.load(bm25_path)
        print(f"    BM25 index has {bm25.size} chunks")
    else:
        print(f"    [!] BM25 index not found at {bm25_path}. Run ingestion first or proceeding with empty index.")
    
    print("\n[3] Loading embedding model...")
    embedder = ChunkEmbedder()
    
    print("\n[4] Loading reranker...")
    reranker = Reranker()
    
    # Create hybrid retriever
    retriever = HybridRetriever(vector_store, bm25, embedder)
    
    # Test queries (domain-specific + out-of-domain negative test)
    test_queries = [
        "What is the computational complexity of standard self-attention with respect to sequence length?",
        "How does Reciprocal Rank Fusion combine dense and sparse rankings?",
        "What speedup does Grover's algorithm provide for searching an unsorted database?",
        "What is the recipe for chocolate chip cookies?",  # Out-of-domain negative test
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Hybrid retrieval
        candidates = retriever.retrieve(query)
        print(f"\nHybrid retrieval returned {len(candidates)} candidates")
        
        # Reranking
        reranked = reranker.rerank(query, candidates)
        print(f"After reranking: {len(reranked)} candidates above threshold")
        
        if not reranked:
            print("  -> No relevant context found.")
            continue
        
        for i, result in enumerate(reranked):
            score = result.get('rerank_score', result.get('rrf_score', 0.0))
            print(f"\n  [{i+1}] Score: {score:.4f}")
            print(f"      Source: {result['metadata'].get('source_file', 'unknown')}")
            print(f"      Section: {result['metadata'].get('section_breadcrumb', 'N/A')}")
            print(f"      Preview: {result['text'][:150]}...")
    
    print("\n" + "="*60)
    print("Retrieval test complete!")


if __name__ == "__main__":
    main()
