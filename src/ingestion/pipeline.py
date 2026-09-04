import pickle
import time
from pathlib import Path

from src.config import settings
from src.ingestion.loader import load_corpus
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.metadata import Chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.logging_config import get_logger

logger = get_logger(__name__)


def get_indexed_hashes(vector_store: VectorStore) -> set[str]:
    """Get the set of file hashes already indexed in the vector store."""
    try:
        collection = vector_store.collection
        # Get all unique file hashes from metadata
        results = collection.get(include=["metadatas"], limit=1)
        if not results["metadatas"]:
            return set()
        
        # Fetch all metadatas in batches to extract file hashes
        count = collection.count()
        if count == 0:
            return set()
            
        hashes = set()
        batch_size = 500
        all_results = collection.get(include=["metadatas"])
        for meta in all_results["metadatas"]:
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])
        return hashes
    except Exception as e:
        logger.warning("failed_to_get_indexed_hashes", error=str(e))
        return set()


def run_ingestion_pipeline(
    corpus_dir: str | None = None,
    force_reindex: bool = False,
) -> dict:
    """Run the full ingestion pipeline: load → chunk → embed → index.
    
    Args:
        corpus_dir: Path to corpus directory. Uses config default if None.
        force_reindex: If True, reindex all documents regardless of existing index.
    
    Returns:
        Dict with ingestion statistics.
    """
    corpus_dir = corpus_dir or settings.corpus_dir
    t_start = time.perf_counter()
    
    # 1. Load documents
    logger.info("ingestion_started", corpus_dir=corpus_dir)
    documents = load_corpus(corpus_dir)
    if not documents:
        logger.warning("no_documents_found", corpus_dir=corpus_dir)
        return {"status": "no_documents", "documents": 0, "chunks": 0}
    
    t_load = time.perf_counter()
    
    # 2. Initialize stores
    vector_store = VectorStore()
    
    # 3. Check existing index for deduplication
    if not force_reindex:
        indexed_hashes = get_indexed_hashes(vector_store)
        documents = [
            (text, doc_info) for text, doc_info in documents
            if doc_info.file_hash not in indexed_hashes
        ]
        if not documents:
            logger.info("all_documents_already_indexed")
            return {"status": "already_indexed", "documents": 0, "chunks": 0}
        logger.info("new_documents_to_index", count=len(documents))
    
    # 4. Chunk all documents
    all_chunks: list[Chunk] = []
    for text, doc_info in documents:
        chunks = chunk_document(text, doc_info)
        all_chunks.extend(chunks)
    
    t_chunk = time.perf_counter()
    logger.info("chunking_complete", total_chunks=len(all_chunks))
    
    if not all_chunks:
        logger.warning("no_chunks_produced")
        return {"status": "no_chunks", "documents": len(documents), "chunks": 0}
    
    # 5. Embed all chunks
    embedder = ChunkEmbedder()
    embeddings = embedder.embed_chunks(all_chunks)
    t_embed = time.perf_counter()
    
    # 6. Store in ChromaDB
    vector_store.add_chunks(all_chunks, embeddings)
    t_store = time.perf_counter()
    
    # 7. Build BM25 index (over ALL chunks in the collection, not just new ones)
    # Reload all chunk texts from ChromaDB for BM25
    all_stored = vector_store.get_all_chunks()
    bm25_index = BM25Index()
    bm25_index.build(all_stored)
    bm25_index.save(str(settings.bm25_pickle_path))
    t_bm25 = time.perf_counter()
    
    stats = {
        "status": "success",
        "documents": len(documents),
        "chunks": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "load_time_s": round(t_load - t_start, 2),
        "chunk_time_s": round(t_chunk - t_load, 2),
        "embed_time_s": round(t_embed - t_chunk, 2),
        "store_time_s": round(t_store - t_embed, 2),
        "bm25_time_s": round(t_bm25 - t_store, 2),
        "total_time_s": round(t_bm25 - t_start, 2),
    }
    
    logger.info("ingestion_complete", **stats)
    return stats
