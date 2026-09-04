"""Pydantic schemas for the RAG chatbot API."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from src.config import settings


class ChatRequest(BaseModel):
    """Request payload for /chat."""
    query: str = Field(..., min_length=1, description="User query text")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace only.")
        if len(trimmed) > settings.max_query_length:
            raise ValueError(f"Query length exceeds maximum of {settings.max_query_length} characters.")
        return trimmed


class Citation(BaseModel):
    """Structured citation metadata for source chunks cited by the LLM."""
    chunk_id: str
    source_file: str
    page: Optional[int] = None
    section: Optional[str] = None
    text_preview: str
    score: float
    citation_number: int


class ChatResponse(BaseModel):
    """Non-streaming response structure for /chat."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class HealthComponent(BaseModel):
    """Health status for a single system component."""
    status: str  # "healthy" | "unhealthy" | "degraded"
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall system health status."""
    status: str  # "ok" | "degraded" | "unhealthy"
    components: dict[str, HealthComponent]
