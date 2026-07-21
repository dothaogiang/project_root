"""
Delete and recreate the document chunk collection.

This is intentionally guarded because it removes indexed content from Qdrant.
Run it only when you explicitly want to rebuild chunks from scratch:

    python -m rag.jobs.reset_chunks --yes
    python -m rag.jobs.sync_job
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient

from rag.config.rag_config import rag_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete the Qdrant document chunk collection.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of the configured chunk collection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    collection = rag_config.COLLECTION_CHUNKS

    if not args.yes:
        print(f"Refusing to delete collection '{collection}' without --yes.")
        print("Run: python -m rag.jobs.reset_chunks --yes")
        return 2

    client = QdrantClient(url=rag_config.QDRANT_URL, api_key=rag_config.QDRANT_API_KEY)

    if client.collection_exists(collection):
        before = client.count(collection_name=collection).count
        client.delete_collection(collection_name=collection)
        print(f"Deleted collection '{collection}' ({before} points).")
    else:
        print(f"Collection '{collection}' does not exist; nothing to delete.")

    print("Now run: python -m rag.jobs.sync_job")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
