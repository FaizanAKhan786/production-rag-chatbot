"""CLI script to run evaluation harness and report retrieval recall, groundedness, and latency.

Usage:
    python scripts/run_eval.py [--no-llm] [--output report.json]
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging
from src.eval.harness import EvaluationHarness


def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluation benchmark.")
    parser.add_argument(
        "--qa-file",
        type=str,
        default=None,
        help="Path to QA pairs JSON file (default: src/eval/fixtures/qa_pairs.json)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation step (evaluates retrieval and reranking only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 65)
    print("RAG Chatbot — Production Evaluation Harness")
    print("=" * 65)

    harness = EvaluationHarness(
        qa_pairs_path=args.qa_file,
        run_generation=not args.no_llm,
    )

    print("\nRunning benchmark across held-out queries...")
    report = harness.run()

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Test Queries:        {report['total_queries']}")
    print(f"Mean Recall@1:             {report['mean_recall@1'] * 100:.1f}%")
    print(f"Mean Recall@3:             {report['mean_recall@3'] * 100:.1f}%")
    print(f"Mean Recall@5:             {report['mean_recall@5'] * 100:.1f}%")
    print(f"Keyword Coverage:          {report['mean_keyword_recall'] * 100:.1f}%")
    print(f"Context Grounded Ratio:    {report['mean_grounded_ratio'] * 100:.1f}%")
    print("-" * 65)
    print(f"Retrieval Latency (p50):   {report['retrieval_latency']['p50_ms']} ms")
    print(f"Retrieval Latency (p95):   {report['retrieval_latency']['p95_ms']} ms")
    print(f"Reranking Latency (p50):   {report['rerank_latency']['p50_ms']} ms")
    print(f"Reranking Latency (p95):   {report['rerank_latency']['p95_ms']} ms")

    if "avg_ms" in report["llm_latency"]:
        print(f"LLM Latency (avg):         {report['llm_latency']['avg_ms']} ms")

    print("=" * 65)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()
