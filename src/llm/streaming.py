"""Async streaming utilities for the RAG chatbot."""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from src.config import settings
from src.llm.client import OllamaClient
from src.llm.prompting import (
    assemble_context,
    build_messages,
    build_citation_objects,
    NO_CONTEXT_RESPONSE,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


async def stream_rag_response(
    query: str,
    scored_chunks: list[dict],
    llm_client: OllamaClient,
) -> AsyncIterator[dict]:
    """Stream a RAG response as a series of SSE-compatible events.
    
    Yields dicts that should be serialized as SSE data:
    - {"event": "token", "data": "..."} for each token
    - {"event": "done", "data": {"citations": [...], "timings": {...}}} on completion
    - {"event": "error", "data": "error message"} on failure
    
    Args:
        query: User's question.
        scored_chunks: Reranked chunks from the retrieval pipeline.
        llm_client: The LLM client to use for generation.
    """
    t_start = time.perf_counter()
    
    # Handle no-context case
    if not scored_chunks:
        # Stream the "no context" response token by token for consistent UX
        words = NO_CONTEXT_RESPONSE.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield {"event": "token", "data": token}
        
        yield {
            "event": "done",
            "data": json.dumps({
                "citations": [],
                "timings": {
                    "llm_total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                },
                "no_context": True,
            }),
        }
        return
    
    # Assemble context and build prompt
    context = assemble_context(scored_chunks)
    messages = build_messages(query, context)
    
    # Stream LLM response
    full_answer = ""
    
    try:
        async for token in llm_client.generate_stream(messages):
            full_answer += token
            yield {"event": "token", "data": token}
        
        t_end = time.perf_counter()
        
        # Build citations from the completed answer
        citations = build_citation_objects(full_answer, scored_chunks)
        
        yield {
            "event": "done",
            "data": json.dumps({
                "citations": citations,
                "timings": {
                    "llm_total_ms": round((t_end - t_start) * 1000, 1),
                },
                "no_context": False,
            }),
        }
        
    except Exception as e:
        logger.warning("llm_offline_fallback", error=str(e), error_type=type(e).__name__)
        
        # Graceful degradation: inform user that Ollama is offline and stream top grounded passages
        offline_msg = (
            "⚠️ **LLM Backend Offline**: The local Ollama service at `" + settings.ollama_base_url + "` is not running.\n\n"
            "*To enable generative answers, start Ollama: `ollama run " + settings.ollama_model + "`*\n\n"
            "**Retrieved & Reranked Grounding Passages:**\n"
        )
        for word in offline_msg.split(" "):
            yield {"event": "token", "data": word + " "}
            
        for i, chunk in enumerate(scored_chunks[:3]):
            meta = chunk.get("metadata", {})
            src = meta.get("source_file", "document")
            sec = meta.get("section_breadcrumb", "")
            excerpt = f"\n\n**[{i+1}] Source: `{src}` {sec}**:\n> {chunk.get('text', '').strip()[:350]}...\n"
            for word in excerpt.split(" "):
                yield {"event": "token", "data": word + " "}

        t_end = time.perf_counter()
        citations = build_citation_objects("[1] [2] [3]", scored_chunks)
        yield {
            "event": "done",
            "data": json.dumps({
                "citations": citations,
                "timings": {
                    "llm_total_ms": round((t_end - t_start) * 1000, 1),
                },
                "no_context": False,
                "offline_fallback": True,
            }),
        }
