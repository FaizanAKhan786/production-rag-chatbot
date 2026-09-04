"""Chat endpoint supporting SSE streaming of answers with grounding citations."""
import time
from typing import AsyncIterator
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest
from src.llm.streaming import stream_rag_response
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Chat"])


@router.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    """Executes hybrid retrieval, reranking, and streams LLM answer with citations via SSE."""
    state = request.app.state
    query = body.query

    async def event_generator() -> AsyncIterator[str]:
        t_start = time.perf_counter()

        # Step 1: Hybrid Retrieval
        t_retrieval_start = time.perf_counter()
        try:
            candidates = state.hybrid_retriever.retrieve(query)
        except Exception as e:
            logger.error("retrieval_failed", error=str(e))
            candidates = []
        t_retrieval_end = time.perf_counter()
        retrieval_ms = round((t_retrieval_end - t_retrieval_start) * 1000, 1)

        # Step 2: Cross-Encoder Reranking (top 15 candidates to optimize throughput)
        t_rerank_start = time.perf_counter()
        try:
            scored_chunks = state.reranker.rerank(query, candidates[:15])
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            scored_chunks = candidates[:6]
        t_rerank_end = time.perf_counter()
        rerank_ms = round((t_rerank_end - t_rerank_start) * 1000, 1)

        retrieved_chunk_ids = [c.get("chunk_id", "") for c in scored_chunks]

        logger.info(
            "query_retrieval_pipeline_complete",
            query=query[:60],
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_candidates=len(candidates),
            filtered_chunks=len(scored_chunks),
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

        try:
            async for item in stream_rag_response(
                query=query,
                scored_chunks=scored_chunks,
                llm_client=state.llm_client,
            ):
                if await request.is_disconnected():
                    logger.info("client_disconnected_stream_aborted")
                    break

                event_type = item.get("event", "token")
                data_content = item.get("data", "")
                # Format standard SSE
                yield f"event: {event_type}\ndata: {data_content}\n\n"
        except Exception as exc:
            logger.error("sse_generation_exception", error=str(exc))
            yield f"event: error\ndata: {str(exc)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
