from pydantic import BaseModel
from typing import Optional
import hashlib


class DocumentInfo(BaseModel):
    """Metadata about a source document."""
    file_path: str
    file_name: str
    file_hash: str  # SHA-256 of file content for deduplication
    file_type: str  # "pdf", "txt", "md"


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk for traceability."""
    chunk_id: str           # unique id: f"{file_hash}_{chunk_index}"
    source_file: str        # original filename
    file_hash: str          # for dedup
    chunk_index: int        # position within document
    page_num: Optional[int] = None
    header_1: Optional[str] = None
    header_2: Optional[str] = None
    header_3: Optional[str] = None
    section_breadcrumb: str = ""  # e.g. "[Introduction > Background]"
    
    def to_chroma_metadata(self) -> dict:
        """Convert to flat dict for ChromaDB storage (only primitive types)."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "file_hash": self.file_hash,
            "chunk_index": self.chunk_index,
            "page_num": self.page_num or -1,
            "header_1": self.header_1 or "",
            "header_2": self.header_2 or "",
            "header_3": self.header_3 or "",
            "section_breadcrumb": self.section_breadcrumb,
        }


class Chunk(BaseModel):
    """A text chunk with its metadata."""
    text: str
    metadata: ChunkMetadata


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha.update(block)
    return sha.hexdigest()[:16]  # truncated for readability
