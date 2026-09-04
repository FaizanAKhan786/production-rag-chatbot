from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    
    # Embedding
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cuda"
    
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    
    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "rag_chunks"
    
    # Retrieval
    bm25_k_retrieve: int = 40
    vector_k_retrieve: int = 40
    rrf_k: int = 60
    rerank_top_k: int = 6
    similarity_threshold: float = 0.35
    
    # Ingestion
    chunk_size: int = 1000
    chunk_overlap: int = 150
    corpus_dir: str = "./data/corpus"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit_rpm: int = 30
    max_query_length: int = 1000
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"
    
    # Derived paths
    @property
    def bm25_pickle_path(self) -> Path:
        return Path(self.chroma_persist_dir).parent / "bm25_index.pickle"
    
    @property
    def corpus_path(self) -> Path:
        return Path(self.corpus_dir)
    
    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


settings = Settings()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
