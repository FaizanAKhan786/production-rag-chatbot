"""Prompt templates and context assembly for the RAG chatbot."""
import re
from typing import Optional

from src.logging_config import get_logger

logger = get_logger(__name__)

# System prompt for the RAG chatbot
SYSTEM_PROMPT_TEMPLATE = """You are a research assistant. Answer the user's question using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context chunks.
2. Cite sources using [1], [2], etc. matching the chunk numbers below.
3. If the context does not contain enough information to answer, say:
   "I don't have enough information in the available documents to answer this question."
4. Do not use knowledge outside of the provided context.
5. Be concise and precise.
6. If multiple sources support a claim, cite all relevant ones.

Context:
{context}"""

NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the available documents to answer this question. "
    "The retrieval system did not find any relevant context above the confidence threshold."
)

# Patterns to strip from retrieved context to prevent prompt injection
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions?",
    r"(?i)ignore\s+(all\s+)?above\s+instructions?",
    r"(?i)disregard\s+(all\s+)?previous",
    r"(?i)you\s+are\s+now\s+a",
    r"(?i)new\s+instructions?:",
    r"(?i)system\s*:\s*",
    r"(?i)\[system\]",
    r"(?i)\[inst\]",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
]


def sanitize_chunk_text(text: str) -> str:
    """Strip common prompt injection patterns from chunk text.
    
    This is a basic defense-in-depth measure. It won't stop all attacks,
    but it catches common patterns.
    """
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized)
    return sanitized


def format_citation(
    index: int,
    chunk: dict,
) -> str:
    """Format a single chunk as a numbered citation block."""
    metadata = chunk.get("metadata", {})
    source = metadata.get("source_file", "unknown")
    page = metadata.get("page_num", -1)
    breadcrumb = metadata.get("section_breadcrumb", "")
    
    # Build source line
    parts = [f"source: {source}"]
    if page and page > 0:
        parts.append(f"p.{page}")
    if breadcrumb:
        parts.append(f"\u00a7{breadcrumb}")
    source_line = ", ".join(parts)
    
    # Sanitize chunk text
    clean_text = sanitize_chunk_text(chunk.get("text", ""))
    
    return f"[{index}] ({source_line})\n{clean_text}"


def assemble_context(scored_chunks: list[dict]) -> str:
    """Assemble formatted context from scored chunks.
    
    Args:
        scored_chunks: List of chunk dicts with 'text', 'metadata', and scores.
    
    Returns:
        Formatted context string with numbered citations.
    """
    if not scored_chunks:
        return ""
    
    formatted_chunks = [
        format_citation(i + 1, chunk)
        for i, chunk in enumerate(scored_chunks)
    ]
    
    return "\n\n".join(formatted_chunks)


def build_messages(
    query: str,
    context: str,
) -> list[dict]:
    """Build the message list for the LLM.
    
    Args:
        query: User's question.
        context: Formatted context string from assemble_context().
    
    Returns:
        List of message dicts for the LLM API.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def extract_citation_numbers(answer: str) -> list[int]:
    """Extract citation numbers [1], [2], etc. from the LLM's answer."""
    matches = re.findall(r'\[(\d+)\]', answer)
    return sorted(set(int(m) for m in matches))


def build_citation_objects(
    answer: str,
    scored_chunks: list[dict],
) -> list[dict]:
    """Build structured citation objects from the LLM's answer.
    
    Maps [1], [2], etc. in the answer back to the actual chunks.
    Only includes citations that the LLM actually referenced.
    """
    cited_numbers = extract_citation_numbers(answer)
    citations = []
    
    for num in cited_numbers:
        idx = num - 1  # 1-indexed to 0-indexed
        if 0 <= idx < len(scored_chunks):
            chunk = scored_chunks[idx]
            metadata = chunk.get("metadata", {})
            citations.append({
                "chunk_id": chunk.get("chunk_id", metadata.get("chunk_id", "")),
                "source_file": metadata.get("source_file", "unknown"),
                "page": metadata.get("page_num", None),
                "section": metadata.get("section_breadcrumb", ""),
                "text_preview": chunk.get("text", "")[:200],
                "score": chunk.get("rerank_score", chunk.get("rrf_score", 0.0)),
                "citation_number": num,
            })
    
    return citations
