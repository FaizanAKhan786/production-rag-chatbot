# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot using FastAPI, ChromaDB, BM25, and Ollama.

## Architecture

```text
[ User ] --> [ FastAPI Server ]
                    |
                    v
            [ Hybrid Retriever ]
            /                  \
   [ Vector DB (Chroma) ]   [ BM25 Index ]
            \                  /
             \                /
              v              v
           [ Reciprocal Rank Fusion ]
                    |
                    v
             [ Cross-Encoder ] (Reranker)
                    |
                    v
           [ Prompt Generation ]
                    |
                    v
              [ LLM (Ollama) ]
                    |
                    v
             [ User Response ]
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with the `qwen2.5:14b` model installed.
  - Install model: `ollama run qwen2.5:14b`

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd rag-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```
   Or install with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` as needed.

## Usage

1. **Ingest Corpus**: Place PDF or text documents in `./data/corpus` and run the ingestion script.
   ```bash
   ingest
   ```

2. **Start Server**: Run the FastAPI application.
   ```bash
   fastapi dev src/api/main.py
   # Or with Uvicorn directly
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

3. **Open UI**: Open the integrated frontend in your browser at `http://localhost:8000`.

## API Endpoints

- `POST /chat`: Stream chat responses (Server-Sent Events).
- `GET /health`: Health check endpoint.

## Eval Harness

To evaluate the system, run the evaluation script located in `src/eval`.

## Project Structure

- `src/api/`: FastAPI routes and application setup.
- `src/ingestion/`: Document processing, chunking, and metadata extraction.
- `src/retrieval/`: Hybrid retrieval logic (BM25 + Vector) and reranking.
- `src/llm/`: Ollama integration and prompt templates.
- `src/eval/`: Evaluation harness and metrics.
- `data/`: Storage for corpora and ChromaDB files.
- `scripts/`: Utility scripts like `ingest`.
