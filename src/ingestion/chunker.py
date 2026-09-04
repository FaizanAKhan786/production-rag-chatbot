from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings
from src.ingestion.metadata import ChunkMetadata, DocumentInfo, Chunk
from src.logging_config import get_logger

logger = get_logger(__name__)

# Headers to split on for structure-aware chunking
HEADERS_TO_SPLIT = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]


def build_breadcrumb(metadata: dict) -> str:
    """Build a section breadcrumb string from header metadata.
    
    e.g., "[Introduction > Background]" or "[Methods]"
    """
    parts = []
    for key in ["header_1", "header_2", "header_3"]:
        val = metadata.get(key, "")
        if val:
            parts.append(val)
    if not parts:
        return ""
    return "[" + " > ".join(parts) + "]"


def chunk_document(text: str, doc_info: DocumentInfo) -> list[Chunk]:
    """Chunk a document using two-stage structure-aware splitting.
    
    Stage 1: Split on markdown headers (MarkdownHeaderTextSplitter)
    Stage 2: Sub-chunk long sections (RecursiveCharacterTextSplitter)
    Then enrich each chunk with breadcrumb context.
    """
    # Stage 1: Split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,  # keep headers in chunk text for context
    )
    
    try:
        header_docs = header_splitter.split_text(text)
    except Exception as e:
        logger.warning("header_split_failed", file=doc_info.file_name, error=str(e))
        # Fallback: treat entire doc as one section
        from langchain_core.documents import Document
        header_docs = [Document(page_content=text, metadata={})]
    
    # Stage 2: Sub-chunk long sections
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    
    chunks: list[Chunk] = []
    chunk_index = 0
    
    for header_doc in header_docs:
        sub_docs = sub_splitter.split_documents([header_doc])
        
        for sub_doc in sub_docs:
            # Extract header metadata
            h1 = sub_doc.metadata.get("header_1", "") or ""
            h2 = sub_doc.metadata.get("header_2", "") or ""
            h3 = sub_doc.metadata.get("header_3", "") or ""
            
            breadcrumb = build_breadcrumb(sub_doc.metadata)
            
            # Prepend breadcrumb to chunk text for embedding
            chunk_text = sub_doc.page_content
            if breadcrumb:
                chunk_text = f"{breadcrumb}\n{chunk_text}"
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
            
            chunk_id = f"{doc_info.file_hash}_{chunk_index:04d}"
            
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=doc_info.file_name,
                file_hash=doc_info.file_hash,
                chunk_index=chunk_index,
                page_num=None,  # page mapping from pymupdf4llm is approximate
                header_1=h1 if h1 else None,
                header_2=h2 if h2 else None,
                header_3=h3 if h3 else None,
                section_breadcrumb=breadcrumb,
            )
            
            chunks.append(Chunk(text=chunk_text, metadata=meta))
            chunk_index += 1
    
    logger.info(
        "document_chunked",
        file=doc_info.file_name,
        total_chunks=len(chunks),
        avg_chunk_len=sum(len(c.text) for c in chunks) // max(len(chunks), 1),
    )
    return chunks
