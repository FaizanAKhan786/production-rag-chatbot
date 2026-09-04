import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from src.config import settings
from src.ingestion.metadata import Chunk
from src.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB vector store wrapper with batch-safe operations."""
    
    def __init__(self) -> None:
        from pathlib import Path
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._max_batch = min(self._client.get_max_batch_size(), 500)
        logger.info(
            "vector_store_initialized",
            collection=settings.chroma_collection,
            existing_count=self._collection.count(),
            max_batch=self._max_batch,
        )
    
    @property
    def collection(self):
        return self._collection
    
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks to the vector store in batches."""
        ids = [c.metadata.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata.to_chroma_metadata() for c in chunks]
        emb_list = embeddings.tolist()
        
        for i in range(0, len(ids), self._max_batch):
            end = min(i + self._max_batch, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=emb_list[i:end],
                metadatas=metadatas[i:end],
            )
            logger.debug("batch_upserted", batch_start=i, batch_end=end)
        
        logger.info("chunks_stored", count=len(ids), total=self._collection.count())
    
    def query(
        self, embedding: np.ndarray, top_k: int = 40
    ) -> list[dict]:
        """Query the vector store.
        
        Args:
            embedding: Query embedding vector.
            top_k: Number of results.
        
        Returns:
            List of dicts with keys: chunk_id, text, metadata, distance.
            Distance is cosine distance (1 - similarity): 0.0 = identical.
        """
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        
        output = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 1.0,
                })
        
        return output
    
    def get_chunks(self, ids: list[str]) -> list[dict]:
        """Get chunks by IDs, batch-safe."""
        output = []
        for i in range(0, len(ids), self._max_batch):
            batch_ids = ids[i:i + self._max_batch]
            results = self._collection.get(
                ids=batch_ids,
                include=["documents", "metadatas"],
            )
            for j, chunk_id in enumerate(results["ids"]):
                output.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][j] if results["documents"] else "",
                    "metadata": results["metadatas"][j] if results["metadatas"] else {},
                })
        return output
    
    def get_all_chunks(self) -> list[tuple[str, str]]:
        """Get all (chunk_id, text) pairs for BM25 index building."""
        count = self._collection.count()
        if count == 0:
            return []
        
        all_results = self._collection.get(
            include=["documents"],
        )
        
        return list(zip(all_results["ids"], all_results["documents"]))
    
    def health_check(self) -> bool:
        """Check if the vector store is accessible."""
        try:
            self._collection.count()
            return True
        except Exception as e:
            logger.error("vector_store_health_check_failed", error=str(e))
            return False
