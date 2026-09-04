"""Integration tests for BM25 and VectorStore retrieval pipeline."""
import numpy as np
import pytest
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.ingestion.metadata import Chunk, ChunkMetadata


def test_bm25_build_and_query(tmp_path):
    bm25 = BM25Index()
    chunks = [
        ("chunk_1", "Quantum superposition and entanglement in qubits."),
        ("chunk_2", "Transformer models use multi-head self-attention mechanisms."),
        ("chunk_3", "Stochastic gradient descent with adaptive learning rates."),
    ]
    bm25.build(chunks)
    assert bm25.is_built
    assert bm25.size == 3

    # Query for quantum
    results = bm25.query("quantum qubits", top_k=2)
    assert len(results) > 0
    assert results[0][0] == "chunk_1"

    # Query for transformers
    results_tf = bm25.query("attention transformer", top_k=2)
    assert len(results_tf) > 0
    assert results_tf[0][0] == "chunk_2"

    # Test save & load
    save_file = str(tmp_path / "bm25_test.pickle")
    bm25.save(save_file)

    loaded_bm25 = BM25Index()
    loaded_bm25.load(save_file)
    assert loaded_bm25.size == 3
    loaded_res = loaded_bm25.query("quantum qubits", top_k=1)
    assert loaded_res[0][0] == "chunk_1"


def test_vector_store_operations():
    vs = VectorStore()
    assert vs.health_check() is True

    # Test adding chunks with synthetic embeddings
    chunks = [
        Chunk(
            text="First test document discussing optimization.",
            metadata=ChunkMetadata(
                chunk_id="t_001",
                source_file="test.md",
                file_hash="h1",
                chunk_index=0,
            ),
        ),
        Chunk(
            text="Second test document discussing attention.",
            metadata=ChunkMetadata(
                chunk_id="t_002",
                source_file="test.md",
                file_hash="h1",
                chunk_index=1,
            ),
        ),
    ]
    # Synthetic 1024-dim normalized embeddings
    emb = np.zeros((2, 1024), dtype=np.float32)
    emb[0, 0] = 1.0  # orthogonal
    emb[1, 1] = 1.0

    vs.add_chunks(chunks, emb)

    # Query matching first vector
    query_emb = np.zeros((1024,), dtype=np.float32)
    query_emb[0] = 1.0
    res = vs.query(query_emb, top_k=2)

    assert len(res) >= 1
    assert res[0]["chunk_id"] == "t_001"
    assert pytest.approx(res[0]["distance"], abs=1e-3) == 0.0

    # Test get_chunks
    retrieved = vs.get_chunks(["t_001"])
    assert len(retrieved) == 1
    assert retrieved[0]["chunk_id"] == "t_001"
