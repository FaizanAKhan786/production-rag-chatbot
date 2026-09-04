"""CLI script to run the ingestion pipeline.

Usage:
    python scripts/ingest.py --corpus-dir ./data/corpus
    python scripts/ingest.py --force-reindex
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging
from src.ingestion.pipeline import run_ingestion_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG chatbot index.")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Path to the corpus directory (default: from .env)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindex all documents, ignoring existing index",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    print("="*60)
    print("RAG Chatbot — Ingestion Pipeline")
    print("="*60)
    
    stats = run_ingestion_pipeline(
        corpus_dir=args.corpus_dir,
        force_reindex=args.force_reindex,
    )
    
    print("\nIngestion Results:")
    print(json.dumps(stats, indent=2))
    
    if stats["status"] not in ("success", "already_indexed"):
        print(f"\n[!] Ingestion ended with status: {stats['status']}")
        sys.exit(1)
    else:
        print(f"\n[OK] Ingestion complete: status={stats['status']} ({stats['chunks']} new chunks indexed from {stats['documents']} documents)")


if __name__ == "__main__":
    main()
