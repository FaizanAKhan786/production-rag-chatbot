import pymupdf4llm
from pathlib import Path
from typing import Optional
import re

from src.ingestion.metadata import DocumentInfo, compute_file_hash
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> str:
    """Load a PDF file and convert to Markdown using pymupdf4llm.
    
    Handles 2-column layouts, tables (as GFM markdown), and header hierarchy.
    """
    logger.info("loading_pdf", file=file_path)
    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text


def load_text(file_path: str) -> str:
    """Load a plain text or markdown file."""
    logger.info("loading_text", file=file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def strip_references(text: str) -> str:
    """Strip the References/Bibliography section to prevent citation-list noise.
    
    Looks for common reference section headers and removes everything after.
    """
    patterns = [
        r"\n#{1,3}\s*References\s*\n",
        r"\n#{1,3}\s*Bibliography\s*\n",
        r"\n#{1,3}\s*Works Cited\s*\n",
        r"\nReferences\s*\n={2,}",  # underline-style headers
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            logger.info("stripped_references", at_position=match.start())
            return text[:match.start()]
    return text


def load_document(file_path: str) -> tuple[str, DocumentInfo]:
    """Load a single document and return its text content and metadata."""
    path = Path(file_path)
    ext = path.suffix.lower()
    file_hash = compute_file_hash(file_path)
    
    if ext == ".pdf":
        text = load_pdf(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        text = load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Strip references section
    text = strip_references(text)
    
    doc_info = DocumentInfo(
        file_path=str(path.resolve()),
        file_name=path.name,
        file_hash=file_hash,
        file_type=ext.lstrip("."),
    )
    
    logger.info("document_loaded", file=path.name, chars=len(text), type=ext)
    return text, doc_info


def load_corpus(corpus_dir: str) -> list[tuple[str, DocumentInfo]]:
    """Load all supported documents from a directory."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    supported_extensions = {".pdf", ".txt", ".md", ".markdown"}
    documents = []
    
    for file_path in sorted(corpus_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                text, doc_info = load_document(str(file_path))
                if text.strip():
                    documents.append((text, doc_info))
                else:
                    logger.warning("empty_document", file=file_path.name)
            except Exception as e:
                logger.error("document_load_failed", file=file_path.name, error=str(e))
    
    logger.info("corpus_loaded", total_documents=len(documents))
    return documents
