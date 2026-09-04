"""LLM client abstraction with Ollama implementation."""
from __future__ import annotations

import time
from typing import AsyncIterator, Protocol, runtime_checkable

from ollama import AsyncClient as OllamaAsyncClient

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Abstract interface for LLM backends."""
    
    async def generate_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        ...
    
    async def generate(
        self, messages: list[dict], **kwargs
    ) -> str:
        """Generate a complete response."""
        ...
    
    async def health_check(self) -> bool:
        """Check if the LLM backend is reachable."""
        ...


class OllamaClient:
    """Ollama LLM client using the official ollama Python package."""
    
    def __init__(self) -> None:
        self._client = OllamaAsyncClient(host=settings.ollama_base_url)
        self._model = settings.ollama_model
        logger.info("ollama_client_initialized", base_url=settings.ollama_base_url, model=self._model)
    
    async def generate_stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama.
        
        Yields individual token strings as they arrive.
        """
        t_start = time.perf_counter()
        first_token = True
        token_count = 0
        
        try:
            stream = await self._client.chat(
                model=self._model,
                messages=messages,
                stream=True,
                options={
                    "temperature": kwargs.get("temperature", 0.2),
                    "num_predict": kwargs.get("max_tokens", 2048),
                },
            )
            
            async for chunk in stream:
                content = chunk.message.content or ""
                if content:
                    if first_token:
                        t_first = time.perf_counter()
                        logger.info(
                            "llm_first_token",
                            ttft_ms=round((t_first - t_start) * 1000, 1),
                        )
                        first_token = False
                    
                    token_count += 1
                    yield content
                
                # Log final telemetry from the last chunk
                if chunk.done:
                    t_end = time.perf_counter()
                    logger.info(
                        "llm_generation_complete",
                        tokens=token_count,
                        total_ms=round((t_end - t_start) * 1000, 1),
                        eval_count=getattr(chunk, 'eval_count', None),
                    )
        except Exception as e:
            logger.error("llm_stream_error", error=str(e), error_type=type(e).__name__)
            raise
    
    async def generate(
        self, messages: list[dict], **kwargs
    ) -> str:
        """Generate a complete (non-streaming) response."""
        t_start = time.perf_counter()
        
        try:
            response = await self._client.chat(
                model=self._model,
                messages=messages,
                stream=False,
                options={
                    "temperature": kwargs.get("temperature", 0.2),
                    "num_predict": kwargs.get("max_tokens", 2048),
                },
            )
            
            t_end = time.perf_counter()
            content = response.message.content or ""
            
            logger.info(
                "llm_generation_complete",
                tokens=len(content.split()),
                total_ms=round((t_end - t_start) * 1000, 1),
            )
            return content
            
        except Exception as e:
            logger.error("llm_generate_error", error=str(e), error_type=type(e).__name__)
            raise
    
    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            # List models to verify connectivity
            models = await self._client.list()
            model_names = [m.model for m in models.models] if models.models else []
            
            # Check if our configured model is available
            # Ollama model names may have tags like ':latest'
            model_base = self._model.split(':')[0]
            available = any(model_base in name for name in model_names)
            
            if not available:
                logger.warning(
                    "ollama_model_not_found",
                    configured=self._model,
                    available=model_names[:5],
                )
            
            return True  # Server is reachable even if model isn't pulled yet
        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return False
