"""End-to-end API tests for health check, rate limiting, and chat endpoint."""
import pytest
from unittest.mock import AsyncMock, patch
from starlette.testclient import TestClient

from src.api.main import app
from src.api.schemas import HealthResponse


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert "vector_store" in data["components"]


def test_chat_query_validation(client):
    # Empty query
    res = client.post("/chat", json={"query": "   "})
    assert res.status_code == 422

    # Missing query
    res2 = client.post("/chat", json={})
    assert res2.status_code == 422


def test_rate_limiting_enforcement(client):
    # Send rapid requests to exceed rate limit (configured via mock or loop)
    # The client can fire requests; with a small rate limit it should trigger 429
    from src.api.middleware import rate_limiter
    rate_limiter.buckets.clear()
    rate_limiter.capacity = 3.0
    rate_limiter.refill_rate = 0.0  # no refill during test

    # 3 allowed
    for _ in range(3):
        res = client.post("/chat", json={"query": "test query"})
        assert res.status_code == 200

    # 4th request must be rejected with 429
    res_blocked = client.post("/chat", json={"query": "test query"})
    assert res_blocked.status_code == 429
    assert res_blocked.json()["error"] == "Too Many Requests"

    # Reset rate limiter
    rate_limiter.capacity = 100.0
    rate_limiter.refill_rate = 100.0 / 60.0
    rate_limiter.buckets.clear()


@pytest.mark.asyncio
async def test_chat_stream_mocked(client):
    # Mock LLM streaming to verify SSE event structure
    async def mock_stream(messages, **kwargs):
        yield "Deep"
        yield " learning"
        yield " uses [1]."

    mock_llm = AsyncMock()
    mock_llm.generate_stream = mock_stream
    mock_llm.health_check.return_value = True

    app.state.llm_client = mock_llm

    res = client.post("/chat", json={"query": "What does deep learning use?"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    body = res.text
    assert "event: token" in body
    assert "event: done" in body
