"""Health check endpoint verifying vector store, reranker, and LLM backend."""
from fastapi import APIRouter, Request
from src.api.schemas import HealthResponse, HealthComponent
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Verifies that vector store, LLM backend, and reranker are reachable."""
    state = request.app.state
    components: dict[str, HealthComponent] = {}
    is_healthy = True

    # 1. Check Vector Store
    try:
        vs_ok = state.vector_store.health_check() if hasattr(state, "vector_store") else False
        if vs_ok:
            count = state.vector_store.collection.count()
            components["vector_store"] = HealthComponent(
                status="healthy",
                detail=f"ChromaDB connected, {count} chunks indexed"
            )
        else:
            components["vector_store"] = HealthComponent(status="unhealthy", detail="Vector store unreachable")
            is_healthy = False
    except Exception as e:
        logger.error("vector_store_health_failed", error=str(e))
        components["vector_store"] = HealthComponent(status="unhealthy", detail=str(e))
        is_healthy = False

    # 2. Check BM25 Index
    try:
        bm25_size = state.bm25_index.size if hasattr(state, "bm25_index") and state.bm25_index.is_built else 0
        components["bm25_index"] = HealthComponent(
            status="healthy" if bm25_size > 0 else "degraded",
            detail=f"BM25 index contains {bm25_size} chunks"
        )
    except Exception as e:
        logger.error("bm25_health_failed", error=str(e))
        components["bm25_index"] = HealthComponent(status="degraded", detail=str(e))

    # 3. Check Reranker
    try:
        reranker_ok = state.reranker.health_check() if hasattr(state, "reranker") else False
        if reranker_ok:
            components["reranker"] = HealthComponent(status="healthy", detail="CrossEncoder model loaded")
        else:
            components["reranker"] = HealthComponent(
                status="degraded",
                detail="Reranker unavailable, running in RRF fallback mode"
            )
    except Exception as e:
        logger.error("reranker_health_failed", error=str(e))
        components["reranker"] = HealthComponent(status="degraded", detail=str(e))

    # 4. Check LLM Backend
    try:
        llm_ok = await state.llm_client.health_check() if hasattr(state, "llm_client") else False
        if llm_ok:
            components["llm_backend"] = HealthComponent(status="healthy", detail="Ollama reachable")
        else:
            components["llm_backend"] = HealthComponent(status="unhealthy", detail="Ollama unreachable")
            is_healthy = False
    except Exception as e:
        logger.error("llm_health_failed", error=str(e))
        components["llm_backend"] = HealthComponent(status="unhealthy", detail=str(e))
        is_healthy = False

    overall_status = "ok" if is_healthy else "degraded"
    return HealthResponse(status=overall_status, components=components)
