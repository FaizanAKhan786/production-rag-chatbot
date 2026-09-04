"""Middleware components for rate limiting, request tracking, and error handling."""
import time
import uuid
from collections import defaultdict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class TokenBucketRateLimiter:
    """In-memory token bucket rate limiter per client IP."""

    def __init__(self, rate_per_minute: int):
        self.capacity = float(rate_per_minute)
        self.refill_rate = float(rate_per_minute) / 60.0  # tokens per second
        self.buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"tokens": self.capacity, "last_updated": time.time()}
        )

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        bucket = self.buckets[client_ip]
        elapsed = now - bucket["last_updated"]
        bucket["last_updated"] = now

        # Add regenerated tokens
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.refill_rate)

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False


rate_limiter = TokenBucketRateLimiter(settings.rate_limit_rpm)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds unique request_id to structlog context and measures request duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit for API endpoints (excluding static files and health)
        if request.url.path.startswith("/chat"):
            if not rate_limiter.is_allowed(client_ip):
                logger.warning("rate_limit_exceeded", client_ip=client_ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "detail": f"Rate limit of {settings.rate_limit_rpm} requests per minute exceeded. Please slow down.",
                    },
                    headers={"Retry-After": "60"},
                )

        t_start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            logger.info("request_finished", status_code=response.status_code, duration_ms=duration_ms)
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            logger.error(
                "unhandled_exception",
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=duration_ms,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": "An unexpected error occurred while processing your request.",
                    "request_id": request_id,
                },
            )
