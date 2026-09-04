import re
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.logging_config import get_logger

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase + extract alphanumeric words.
    
    Must be consistent between indexing and querying.
    """
    return re.findall(r'\w+', text.lower())


class BM25Index:
    """BM25 sparse retrieval index.
    
    Wraps rank_bm25.BM25Okapi with save/load and top-K query support.
    Note: does not support incremental updates — must rebuild on corpus changes.
    """
    
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
    
    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
    
    @property
    def size(self) -> int:
        return len(self._chunk_ids)
    
    def build(self, chunks: list[tuple[str, str]]) -> None:
        """Build the BM25 index from chunks.
        
        Args:
            chunks: List of (chunk_id, chunk_text) tuples.
        """
        if not chunks:
            logger.warning("bm25_build_empty_corpus")
            return
        
        self._chunk_ids = [cid for cid, _ in chunks]
        self._chunk_texts = [text for _, text in chunks]
        
        tokenized_corpus = [tokenize(text) for text in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("bm25_index_built", num_chunks=len(self._chunk_ids))
    
    def query(self, text: str, top_k: int = 40) -> list[tuple[str, float]]:
        """Query the BM25 index.
        
        Args:
            text: Query text.
            top_k: Number of top results to return.
        
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending.
        """
        if not self.is_built:
            raise RuntimeError("BM25 index not built. Call build() first.")
        
        tokenized_query = tokenize(text)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_k = min(top_k, len(self._chunk_ids))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # skip zero-score results
        ]
        
        return results
    
    def save(self, path: str) -> None:
        """Save the BM25 index to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self._chunk_ids,
            "chunk_texts": self._chunk_texts,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("bm25_index_saved", path=path, num_chunks=len(self._chunk_ids))
    
    def load(self, path: str) -> None:
        """Load a BM25 index from a pickle file and rebuild."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._chunk_ids = data["chunk_ids"]
        self._chunk_texts = data["chunk_texts"]
        
        tokenized_corpus = [tokenize(text) for text in self._chunk_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("bm25_index_loaded", path=path, num_chunks=len(self._chunk_ids))
