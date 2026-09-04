"""Unit tests for document chunker and loader."""
from src.ingestion.chunker import chunk_document, build_breadcrumb
from src.ingestion.loader import strip_references
from src.ingestion.metadata import DocumentInfo


def test_build_breadcrumb():
    meta1 = {"header_1": "Introduction"}
    assert build_breadcrumb(meta1) == "[Introduction]"

    meta2 = {"header_1": "Methods", "header_2": "Attention"}
    assert build_breadcrumb(meta2) == "[Methods > Attention]"

    meta3 = {"header_1": "Experiments", "header_2": "Ablation", "header_3": "Heads"}
    assert build_breadcrumb(meta3) == "[Experiments > Ablation > Heads]"

    assert build_breadcrumb({}) == ""


def test_strip_references():
    text = (
        "# Main Content\n"
        "This is the core discussion.\n\n"
        "## References\n"
        "[1] Vaswani et al. Attention is all you need.\n"
        "[2] Devlin et al. BERT."
    )
    stripped = strip_references(text)
    assert "This is the core discussion." in stripped
    assert "Vaswani et al" not in stripped
    assert "## References" not in stripped


def test_chunk_document_structure_and_breadcrumbs():
    doc_info = DocumentInfo(
        file_path="paper.md",
        file_name="paper.md",
        file_hash="test1234",
        file_type="md",
    )
    content = (
        "# Introduction\n\n"
        "Transformers are powerful neural architectures for representation learning.\n\n"
        "## Multi-Head Attention\n\n"
        "Multi-head attention projects inputs into several representation subspaces.\n\n"
        "## Computational Cost\n\n"
        "The quadratic complexity in sequence length presents memory bottlenecks."
    )

    chunks = chunk_document(content, doc_info)
    assert len(chunks) >= 3

    # Verify each chunk has metadata and breadcrumbs
    for i, c in enumerate(chunks):
        assert c.metadata.chunk_id.startswith("test1234_")
        assert c.metadata.chunk_index == i
        assert c.metadata.source_file == "paper.md"
        assert c.metadata.file_hash == "test1234"
        assert len(c.text) > 0

    # Verify breadcrumb prepending
    chunk_texts = [c.text for c in chunks]
    assert any("[Introduction]" in t for t in chunk_texts)
    assert any("Multi-Head Attention" in t for t in chunk_texts)
