# Multi-stage Dockerfile for Production RAG Chatbot
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies (e.g. curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 raguser && \
    mkdir -p /app/data && \
    chown -R raguser:raguser /app

# Copy application files
COPY src/ ./src/
COPY static/ ./static/
COPY scripts/ ./scripts/
COPY pyproject.toml .env.example ./

# Copy .env if present, otherwise fallback to .env.example
RUN cp .env.example .env && chown -R raguser:raguser /app

USER raguser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
