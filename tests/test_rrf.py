"""Unit tests for Reciprocal Rank Fusion (RRF)."""
import pytest
from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_formula_correctness():
    """Validates mathematical correctness of RRF scoring with known values."""
    # List 1: ["doc_a", "doc_b"]
    # List 2: ["doc_b", "doc_c"]
    # For k = 60:
    # doc_a: rank 1 in L1 -> 1 / (60 + 1) = 1/61 ≈ 0.016393
    # doc_b: rank 2 in L1 -> 1 / (60 + 2) = 1/62 ≈ 0.016129
    #        rank 1 in L2 -> 1 / (60 + 1) = 1/61 ≈ 0.016393
    #        total doc_b = 1/61 + 1/62 ≈ 0.032522
    # doc_c: rank 2 in L2 -> 1 / (60 + 2) = 1/62 ≈ 0.016129

    list1 = ["doc_a", "doc_b"]
    list2 = ["doc_b", "doc_c"]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    fused_dict = dict(fused)

    expected_b = (1.0 / 61.0) + (1.0 / 62.0)
    expected_a = 1.0 / 61.0
    expected_c = 1.0 / 62.0

    assert pytest.approx(fused_dict["doc_b"], abs=1e-6) == expected_b
    assert pytest.approx(fused_dict["doc_a"], abs=1e-6) == expected_a
    assert pytest.approx(fused_dict["doc_c"], abs=1e-6) == expected_c

    # Ensure doc_b is ranked #1 because it appeared in both lists
    assert fused[0][0] == "doc_b"
    assert fused[1][0] == "doc_a"
    assert fused[2][0] == "doc_c"


def test_rrf_disjoint_lists():
    """Handles completely disjoint result lists."""
    list1 = ["doc_1", "doc_2"]
    list2 = ["doc_3", "doc_4"]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 4
    # rank 1 items should have identical score: 1/61
    fused_dict = dict(fused)
    assert pytest.approx(fused_dict["doc_1"]) == fused_dict["doc_3"]


def test_rrf_empty_lists():
    """Handles empty lists gracefully."""
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[]], k=60) == []
    assert len(reciprocal_rank_fusion([["doc_1"], []], k=60)) == 1
