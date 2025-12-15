"""Entry point for running memory as a module: python -m memory"""

import argparse
import asyncio
import os
import pathlib
import sys
import time

from qdrant_client import QdrantClient, models

from .memory import FastRAGPipeline
from .models import EmbeddingConfig, QdrantConfig, RAGConfig


def is_localhost(url: str) -> bool:
    """Check if the URL is a localhost connection."""
    return url in ("localhost", "http://localhost:6333", "http://127.0.0.1:6333")


def get_api_key(args: argparse.Namespace, url: str) -> str | None:
    """Get API key from args or environment (CLI-only).

    API key is optional for localhost connections.
    """
    api_key = args.api_key or os.environ.get("QDRANT_API_KEY")
    if not api_key and not is_localhost(url):
        print("Error: --api-key is required or set QDRANT_API_KEY environment variable")
        sys.exit(1)
    return api_key


def get_url(args: argparse.Namespace) -> str:
    """Get Qdrant URL from args or environment (CLI-only).

    Handles 'localhost' shortcut by expanding to full URL.
    """
    url = args.url or os.environ.get("QDRANT_URL")
    if not url:
        print("Error: --url is required or set QDRANT_URL environment variable")
        sys.exit(1)
    # Expand localhost shortcut
    if url == "localhost":
        return "http://localhost:6333"
    return url


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Document ingestion and retrieval CLI for RAG pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents from disk into the vector database",
    )
    ingest_parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to file or directory containing documents",
    )
    ingest_parser.add_argument(
        "--type",
        required=True,
        choices=["markdown", "pdf", "text"],
        help="Document type to ingest",
    )
    ingest_parser.add_argument(
        "--collection",
        required=True,
        help="Name of the Qdrant collection",
    )
    ingest_parser.add_argument(
        "--url",
        help="Qdrant server URL (or set QDRANT_URL env var)",
    )
    ingest_parser.add_argument(
        "--api-key",
        help="Qdrant API key (or set QDRANT_API_KEY env var)",
    )
    ingest_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively",
    )
    ingest_parser.add_argument(
        "--cache-dir",
        default=".cache/fastembed",
        help="Directory to cache embedding models (default: .cache/fastembed)",
    )
    ingest_parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use LLM-powered agentic chunking (slower but more semantic)",
    )
    ingest_parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name (default: BAAI/bge-small-en-v1.5)",
    )
    ingest_parser.add_argument(
        "--embedding-service-url",
        help="URL of remote embedding service (e.g., http://localhost:8000)",
    )

    # query subcommand
    query_parser = subparsers.add_parser(
        "query",
        help="Query the vector database for relevant context",
    )
    query_parser.add_argument(
        "query",
        type=str,
        help="The search query text",
    )
    query_parser.add_argument(
        "--collection",
        required=True,
        help="Name of the Qdrant collection",
    )
    query_parser.add_argument(
        "--url",
        help="Qdrant server URL (or set QDRANT_URL env var)",
    )
    query_parser.add_argument(
        "--api-key",
        help="Qdrant API key (or set QDRANT_API_KEY env var)",
    )
    query_parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of results to return (default: 3)",
    )
    query_parser.add_argument(
        "--cache-dir",
        default=".cache/fastembed",
        help="Directory to cache embedding models (default: .cache/fastembed)",
    )
    query_parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name (default: BAAI/bge-small-en-v1.5)",
    )
    query_parser.add_argument(
        "--embedding-service-url",
        help="URL of remote embedding service (e.g., http://localhost:8000)",
    )

    # migrate subcommand
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Migrate collection from one Qdrant instance to another",
    )
    migrate_parser.add_argument(
        "--source-url",
        required=True,
        help="Source Qdrant server URL (e.g., cloud instance)",
    )
    migrate_parser.add_argument(
        "--source-api-key",
        help="Source Qdrant API key (or set QDRANT_SOURCE_API_KEY env var)",
    )
    migrate_parser.add_argument(
        "--dest-url",
        required=True,
        help="Destination Qdrant server URL (e.g., http://localhost:6333)",
    )
    migrate_parser.add_argument(
        "--dest-api-key",
        help="Destination Qdrant API key (or set QDRANT_DEST_API_KEY env var)",
    )
    migrate_parser.add_argument(
        "--collection",
        required=True,
        help="Name of the collection to migrate",
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of points to migrate per batch (default: 100)",
    )

    return parser


async def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "migrate":
        # Migration doesn't need FastRAGPipeline, uses QdrantClient directly
        source_api_key = args.source_api_key or os.environ.get("QDRANT_SOURCE_API_KEY")
        dest_api_key = args.dest_api_key or os.environ.get("QDRANT_DEST_API_KEY")

        source_client = QdrantClient(url=args.source_url, api_key=source_api_key)
        if args.dest_url == "localhost":
            dest_client = QdrantClient(url="localhost", port=6333)
        else:
            dest_client = QdrantClient(url=args.dest_url, api_key=dest_api_key)

        # Get source collection info
        collection_info = source_client.get_collection(args.collection)
        vector_config = collection_info.config.params.vectors

        print(
            f"Migrating collection '{args.collection}' from {args.source_url} to {args.dest_url}"
        )
        print(f"Vector size: {vector_config.size}, Distance: {vector_config.distance}")

        # Create destination collection with same config
        if dest_client.collection_exists(args.collection):
            dest_client.delete_collection(args.collection)
        dest_client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(
                size=vector_config.size,
                distance=vector_config.distance,
            ),
        )

        # Scroll and migrate points
        offset = None
        total = 0
        st = time.time()

        while True:
            records, offset = source_client.scroll(
                collection_name=args.collection,
                limit=args.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if records:
                # Convert Record objects to PointStruct for upsert
                points = [
                    models.PointStruct(
                        id=record.id,
                        vector=record.vector,
                        payload=record.payload,
                    )
                    for record in records
                ]
                dest_client.upsert(collection_name=args.collection, points=points)
                total += len(points)
                print(f"Migrated {total} points...")

            if offset is None:
                break

        et = time.time()
        print(f"Migration complete: {total} points migrated in {(et - st):.2f}s")
        return

    # For ingest/query commands, use FastRAGPipeline
    url = get_url(args)
    api_key = get_api_key(args, url)

    config = RAGConfig(
        qdrant=QdrantConfig(
            url=url,
            api_key=api_key,
            collection_name=args.collection,
        ),
        embedding=EmbeddingConfig(
            model_name=args.model,
            cache_dir=args.cache_dir,
            embedding_service_url=getattr(args, "embedding_service_url", None),
        ),
    )

    pipeline = FastRAGPipeline(config)

    if args.command == "ingest":
        await pipeline.initialize_collection()
        count = await pipeline.create_and_ingest_documents(
            args.path,
            type=args.type,
            recursive=args.recursive,
            use_agentic_chunker=args.agentic,
        )
        print(f"Ingested {count} chunks into collection '{args.collection}'")

    elif args.command == "query":
        st = time.time()
        result = await pipeline.retrieve_context(args.query, limit=args.limit)
        et = time.time()
        if not result.texts:
            print(f"No results found.")
        else:
            for i, (text, score) in enumerate(zip(result.texts, result.scores), 1):
                print(f"\n[{i}] Score: {score:.4f}")
                print(text)
        print(f"Retrieval took {(et - st):.4f}s")


if __name__ == "__main__":
    asyncio.run(main())
