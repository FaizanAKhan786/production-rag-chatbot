import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings, BGE_QUERY_PREFIX
from src.ingestion.metadata import Chunk
from src.logging_config import get_logger

logger = get_logger(__name__)


class ChunkEmbedder:
    """Embeds text chunks using a SentenceTransformer model."""
    
    def __init__(self) -> None:
        logger.info("loading_embedding_model", model=settings.embedding_model, device=settings.embedding_device)
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info("embedding_model_loaded", dimension=self._dimension)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        """Embed document chunks. Does NOT add query prefix (that's for queries only)."""
        texts = [chunk.text for chunk in chunks]
        logger.info("embedding_chunks", count=len(texts), batch_size=batch_size)
        
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # required for cosine similarity
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )
        
        logger.info("chunks_embedded", count=len(texts), shape=embeddings.shape)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the BGE instruction prefix."""
        prefixed = BGE_QUERY_PREFIX + query
        embedding = self._model.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return embedding
