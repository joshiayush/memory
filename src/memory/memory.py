"""
Real-Time RAG Pipeline for LiveKit Voice Agents.

This module provides a high-performance Retrieval-Augmented Generation (RAG)
pipeline optimized for real-time voice agent applications. It uses Qdrant as
the vector database and FastEmbed for CPU-optimized embeddings.

Key Design Decisions:
    - Synchronous QdrantClient wrapped in asyncio.to_thread() for non-blocking I/O
    - FastEmbed with ONNX Runtime for fast CPU inference (~10-15ms per query)
    - Target retrieval latency: <200ms to avoid perceptible voice delay

Example:
    >>> config = RAGConfig(
    ...     qdrant=QdrantConfig(url="http://localhost:6333"),
    ...     embedding=EmbeddingConfig(),
    ... )
    >>> pipeline = FastRAGPipeline(config)
    >>> await pipeline.initialize_collection()
    >>> await pipeline.ingest_documents(["LiveKit enables real-time communication."])
    >>> results = await pipeline.retrieve_context("What is LiveKit?")
"""

from __future__ import annotations

import asyncio
import pathlib
import uuid
from typing import List

from fastembed import TextEmbedding
from fastembed.common.types import NumpyArray
from qdrant_client import QdrantClient, models
from tidylog import get_logger

from .chunker.agentic.chunker import AgenticChunker
from .content import discover_files, extract_propositions, load_document
from .content.types import DocType
from .models import QdrantConfig, RAGConfig, RetrievalResult

logger = get_logger(__name__)


class FastRAGPipeline:
    """High-performance RAG pipeline optimized for real-time voice agents.

    This pipeline is designed for sub-200ms retrieval latency, making it suitable
    for voice agent applications where response time directly impacts user experience.

    Architecture Notes:
        - Uses synchronous QdrantClient because AsyncQdrantClient doesn't support
          local (:memory:) mode and can have higher latency for remote connections.
        - All blocking operations are wrapped in asyncio.to_thread() to prevent
          blocking the event loop (critical for voice agents to avoid audio glitches).
        - Embedding model is pre-loaded at initialization to avoid cold-start latency.

    Example:
        >>> config = RAGConfig(
        ...     qdrant=QdrantConfig(
        ...         url="https://your-cluster.cloud.qdrant.io:6333",
        ...         api_key="your-api-key",
        ...         collection_name="livekit_docs",
        ...     ),
        ...     embedding=EmbeddingConfig(model_name="BAAI/bge-small-en-v1.5"),
        ... )
        >>> pipeline = FastRAGPipeline(config)
        >>> await pipeline.initialize_collection()
        >>> await pipeline.ingest_documents(documents)
        >>> result = await pipeline.retrieve_context("How do I create a room?")
        >>> print(result.top_result)

    Attributes:
        config: The RAG configuration object.
        client: Qdrant client instance for database operations.
        embedding_model: FastEmbed model instance for generating embeddings.
    """

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the RAG pipeline with the given configuration.

        This constructor pre-loads the embedding model, which may take a few seconds
        on first run (model download) but ensures fast inference during operation.

        Args:
            config: RAG pipeline configuration containing Qdrant and embedding settings.

        Raises:
            ConnectionError: If unable to connect to the Qdrant server.
            ValueError: If the embedding model cannot be loaded.
        """
        self.config = config
        self.client = QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key,
        )

        # Pre-load embedding model to avoid cold-start latency during retrieval
        # FastEmbed uses ONNX Runtime for optimized CPU inference
        # cache_dir ensures models are persisted and not re-downloaded
        self.embedding_model = TextEmbedding(
            model_name=config.embedding.model_name,
            threads=config.embedding.threads,
            cache_dir=config.embedding.cache_dir,
        )

    async def initialize_collection(self) -> bool:
        """Create the vector collection if it doesn't exist.

        This method is idempotent - calling it multiple times is safe.
        The collection is configured with HNSW indexing optimized for
        fast approximate nearest neighbor search.

        Returns:
            True if a new collection was created, False if it already existed.

        Raises:
            ConnectionError: If unable to communicate with Qdrant server.
        """
        # Check if collection already exists
        exists = await asyncio.to_thread(
            self.client.collection_exists,
            self.config.qdrant.collection_name,
        )

        if exists:
            return False

        # Create collection with optimized settings for real-time retrieval
        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=self.config.qdrant.collection_name,
            vectors_config=models.VectorParams(
                size=self.config.qdrant.vector_size,
                distance=self.config.qdrant.distance_metric,
                on_disk=self.config.qdrant.on_disk,
            ),
            # HNSW optimizer settings for balanced speed/accuracy
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=2,  # Fewer segments = faster search
            ),
        )

        logger.info(
            "collection created",
            extra={"collection": self.config.qdrant.collection_name},
        )

        return True

    def _embed_sync(self, texts: List[str]) -> List[NumpyArray]:
        """Generate embeddings synchronously (internal helper).

        This method is called via asyncio.to_thread() to prevent blocking
        the event loop during CPU-intensive embedding computation.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (numpy arrays).
        """
        # FastEmbed.embed() returns a generator, consume it immediately
        return list(self.embedding_model.embed(texts))

    async def ingest_documents(
        self,
        documents: List[str],
    ) -> int:
        """Ingest documents into the vector database.

        Documents are embedded and stored in Qdrant for later retrieval.
        Each document is assigned a unique UUID as its identifier.

        Args:
            documents: List of document texts to ingest.

        Returns:
            Number of documents successfully ingested.

        Raises:
            ValueError: If documents list is empty.
            ConnectionError: If unable to communicate with Qdrant server.

        Example:
            >>> docs = ["LiveKit is open source.", "Agents handle voice AI."]
            >>> count = await pipeline.ingest_documents(docs)
            >>> print(f"Ingested {count} documents")
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Generate embeddings (CPU-bound operation)
        # Running in separate thread prevents blocking the event loop,
        # which is critical for voice agents to avoid audio glitches
        embeddings = await asyncio.to_thread(self._embed_sync, documents)

        # Prepare points for Qdrant upsert
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": document},
            )
            for document, embedding in zip(documents, embeddings)
        ]

        # Upsert to Qdrant (IO-bound operation)
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.config.qdrant.collection_name,
            points=points,
        )

        return len(documents)

    async def create_and_ingest_documents(
        self,
        path: pathlib.Path,
        *,
        type: DocType,
        recursive: bool = False,
    ) -> int:
        """Load, chunk, and ingest documents from disk.

        This method discovers files at the given path, loads their content,
        extracts propositions, semantically chunks them using the AgenticChunker,
        and ingests the resulting chunks into the vector database.

        Args:
            path: Path to a file or directory containing documents.
            type: The document type to load (markdown, pdf, or text).
            recursive: If True and path is a directory, search subdirectories.

        Returns:
            Number of chunks successfully ingested.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If no documents are found or all documents are empty.

        Example:
            >>> count = await pipeline.create_and_ingest_documents(
            ...     Path("./docs"),
            ...     type="markdown",
            ...     recursive=True,
            ... )
            >>> print(f"Ingested {count} chunks")
        """
        # Discover files matching the document type
        files = await discover_files(path, type, recursive=recursive)

        if not files:
            raise ValueError(f"No {type} files found at {path}")

        logger.info(
            "discovered files",
            extra={"count": len(files), "type": type, "path": str(path)},
        )

        # Load content from all files
        all_propositions: List[str] = []
        for file_path in files:
            content = await load_document(file_path, type)
            if content.strip():
                propositions = extract_propositions(content)
                all_propositions.extend(propositions)

        if not all_propositions:
            raise ValueError("No content extracted from documents")

        logger.info(
            "extracted propositions",
            extra={"count": len(all_propositions)},
        )

        # Chunk propositions semantically
        chunker = AgenticChunker(generate_new_metadata=True)
        chunker.add_propositions(all_propositions)
        documents = chunker.to_documents()

        logger.info(
            "chunking complete",
            extra={"chunks": len(documents)},
        )

        # Ingest chunks into vector database
        count = await self.ingest_documents(documents)

        return count

    async def retrieve_context(
        self,
        query: str,
        limit: int = 3,
    ) -> RetrievalResult:
        """Retrieve relevant context for a query.

        This is the primary method for RAG retrieval. It embeds the query,
        searches for similar documents, and returns results with latency metrics.

        Performance Target:
            Total latency should be <200ms for voice agent applications.
            Typical breakdown: ~15ms embedding + ~50-150ms search.

        Args:
            query: The search query text.
            limit: Maximum number of results to return (default: 3).

        Returns:
            RetrievalResult containing matching documents, scores, and latency metrics.

        Raises:
            ValueError: If query is empty.
            ConnectionError: If unable to communicate with Qdrant server.

        Example:
            >>> result = await pipeline.retrieve_context("What is LiveKit?")
            >>> if result.top_result:
            ...     print(f"Found: {result.top_result}")
            ...     print(f"Latency: {result.total_latency_ms:.2f}ms")
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Embed the query (CPU-bound)
        # Must run in thread to avoid blocking WebSocket heartbeat in voice agents
        query_embeddings = await asyncio.to_thread(self._embed_sync, [query])
        query_vector = query_embeddings[0]

        # Search for similar documents (IO-bound)
        search_result = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.config.qdrant.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_vectors=False,  # Don't return vectors to reduce payload size
        )

        # Extract results
        texts = [hit.payload["text"] for hit in search_result.points]
        scores = [hit.score for hit in search_result.points]

        return RetrievalResult(
            texts=texts,
            scores=scores,
        )

    async def delete_collection(self) -> bool:
        """Delete the vector collection.

        Use with caution - this permanently removes all stored documents.

        Returns:
            True if collection was deleted, False if it didn't exist.
        """
        exists = await asyncio.to_thread(
            self.client.collection_exists,
            self.config.qdrant.collection_name,
        )

        if not exists:
            return False

        await asyncio.to_thread(
            self.client.delete_collection,
            collection_name=self.config.qdrant.collection_name,
        )

        logger.info(
            "collection deleted",
            extra={"collection": self.config.qdrant.collection_name},
        )

        return True
