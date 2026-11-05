# Production RAG Chatbot | Quality & reproducibility

> **Status rule:** Check the [live GitHub Actions history](https://github.com/FaizanAKhan786/production-rag-chatbot/actions) for real workflow conclusions. This document does not invent benchmark scores or imply every feature was tested.

## Automated verification

Python syntax compilation and selected deterministic RRF/chunker tests; XML unit test report attached to GitHub Actions run.

- [View the CI workflow](https://github.com/FaizanAKhan786/production-rag-chatbot/actions/workflows/ci.yml)
- Any failing run is an unresolved result until corrected; a green syntax/unit-check workflow is not proof of deployment readiness.

## Reproduce the checks

Run these commands from the **repository root**, unless the commands change directories:

```bash
pip install -e ".[dev]"
python -m pytest tests/test_rrf.py tests/test_chunker.py -q
# Requires an indexed corpus and retrieval/model assets:
python scripts/run_eval.py --no-llm --output retrieval_eval.json
```

## How to interpret the results

**Relevant metrics:** Held-out retrieval recall, keyword coverage, context grounded ratio, reranking latency and optional generation latency.

**Requirements and limitations:** Before publishing metrics, record corpus hashes, QA fixtures, embedding model, reranker, Ollama model and hardware. CI unit tests do not benchmark end-to-end quality.

Before comparing results, save the commit SHA, command, dataset and split (including seed), model revision, dependency environment, hardware/CPU/GPU, and the raw result file. Use the same evaluation set for all compared runs and distinguish **synthetic tests** from **real-world measurements**.

## Publication policy

Only publish numeric scores alongside the run configuration and raw evidence. A threshold, code assertion or example report must **not** be presented as a measured production result.

[Project README](../README.md) · [GitHub Actions](https://github.com/FaizanAKhan786/production-rag-chatbot/actions)
