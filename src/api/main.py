"""FastAPI main application entrypoint for RAG Chatbot."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.logging_config import setup_logging, get_logger
from src.api.middleware import RequestContextMiddleware
from src.api.routes import health, chat
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index
from src.ingestion.embedder import ChunkEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.llm.client import OllamaClient

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown resource management."""
    setup_logging()
    logger.info("service_starting_up")

    # 1. Initialize Vector Store
    vector_store = VectorStore()
    app.state.vector_store = vector_store

    # 2. Initialize BM25 Index
    bm25_index = BM25Index()
    bm25_path = str(settings.bm25_pickle_path)
    if Path(bm25_path).exists():
        try:
            bm25_index.load(bm25_path)
            logger.info("bm25_index_loaded_on_startup", size=bm25_index.size)
        except Exception as e:
            logger.warning("bm25_startup_load_failed", error=str(e))
    else:
        logger.info("bm25_no_existing_index_found", path=bm25_path)
    app.state.bm25_index = bm25_index

    # 3. Initialize Embedder & Reranker
    embedder = ChunkEmbedder()
    app.state.embedder = embedder

    reranker = Reranker()
    app.state.reranker = reranker

    # 4. Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedder=embedder,
    )
    app.state.hybrid_retriever = hybrid_retriever

    # 5. Initialize LLM Client
    llm_client = OllamaClient()
    app.state.llm_client = llm_client

    logger.info("service_startup_completed")
    yield

    logger.info("service_shutting_down")


app = FastAPI(
    title="Production RAG Chatbot",
    description="High-accuracy academic research paper RAG with hybrid search and reranking",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware (Rate limiting, logging context, error handling)
app.add_middleware(RequestContextMiddleware)

# Routes
app.include_router(health.router)
app.include_router(chat.router)

# Static files and UI
static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index_page():
        return FileResponse(str(static_dir / "index.html"))
