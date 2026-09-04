"""Pytest fixtures and configuration."""
import os
import pytest
from pathlib import Path

# Ensure testing environment variables are set before imports
os.environ["CHROMA_PERSIST_DIR"] = "./data/test_chroma_db"
os.environ["CHROMA_COLLECTION"] = "test_collection"
os.environ["EMBEDDING_DEVICE"] = "cpu"
os.environ["RERANKER_DEVICE"] = "cpu"
os.environ["RATE_LIMIT_RPM"] = "100"
os.environ["LOG_FORMAT"] = "console"

from src.ingestion.metadata import DocumentInfo, ChunkMetadata, Chunk


@pytest.fixture
def sample_doc_info() -> DocumentInfo:
    return DocumentInfo(
        file_path="/data/sample_paper.pdf",
        file_name="sample_paper.pdf",
        file_hash="abcd1234efgh5678",
        file_type="pdf",
    )


@pytest.fixture
def sample_chunks(sample_doc_info) -> list[Chunk]:
    return [
        Chunk(
            text="[Introduction]\nDeep neural networks require substantial parameter tuning.",
            metadata=ChunkMetadata(
                chunk_id=f"{sample_doc_info.file_hash}_0000",
                source_file=sample_doc_info.file_name,
                file_hash=sample_doc_info.file_hash,
                chunk_index=0,
                page_num=1,
                header_1="Introduction",
                section_breadcrumb="[Introduction]",
            ),
        ),
        Chunk(
            text="[Methods > Optimization]\nStochastic gradient descent with momentum converges reliably.",
            metadata=ChunkMetadata(
                chunk_id=f"{sample_doc_info.file_hash}_0001",
                source_file=sample_doc_info.file_name,
                file_hash=sample_doc_info.file_hash,
                chunk_index=1,
                page_num=2,
                header_1="Methods",
                header_2="Optimization",
                section_breadcrumb="[Methods > Optimization]",
            ),
        ),
    ]
