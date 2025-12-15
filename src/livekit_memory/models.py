from dataclasses import dataclass, field
from typing import Optional, List

from qdrant_client import models

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database connection.

    Attributes:
        url: Qdrant server URL. Use ":memory:" for in-memory storage (testing only),
            "http://localhost:6333" for local Docker, or cloud URL for production.
        api_key: API key for Qdrant Cloud authentication. None for local instances.
        collection_name: Name of the vector collection to use.
        vector_size: Dimensionality of embedding vectors. Must match the embedding model.
            Common sizes: 384 (bge-small), 768 (bge-base), 1024 (bge-large).
        distance_metric: Distance metric for similarity search.
            COSINE is recommended for normalized embeddings.
        on_disk: Whether to store vectors on disk. True reduces memory usage
            but may slightly increase latency.
    """

    url: str
    api_key: Optional[str]
    collection_name: str
    vector_size: int = 384
    distance_metric: models.Distance = models.Distance.COSINE
    on_disk: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model.

    Attributes:
        model_name: HuggingFace model identifier. FastEmbed downloads and caches
            the ONNX-optimized version automatically.
            Recommended models:
                - "sentence-transformers/all-MiniLM-L6-v2" (384 dims, default)
                - "BAAI/bge-small-en-v1.5" (384 dims, fastest, good quality)
                - "BAAI/bge-base-en-v1.5" (768 dims, balanced)
                - "BAAI/bge-large-en-v1.5" (1024 dims, highest quality)
        threads: Number of CPU threads for ONNX Runtime inference.
            None uses all available cores (recommended for dedicated servers).
            Set to a specific number to limit CPU usage in shared environments.
        cache_dir: Directory to cache downloaded models. If None, uses the
            default .cache/fastembed directory. Set this to persist models
            across runs and avoid re-downloading.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threads: Optional[int] = None
    cache_dir: Optional[str] = None


@dataclass
class RAGConfig:
    """Combined configuration for the RAG pipeline.

    Attributes:
        qdrant: Qdrant database configuration.
        embedding: Embedding model configuration.
    """

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class RetrievalResult:
    """Result from a single retrieval operation.

    Attributes:
        texts: List of retrieved document texts, ordered by relevance (most relevant first).
        scores: Similarity scores corresponding to each text. Higher is more similar.
            For cosine distance, scores range from 0 (dissimilar) to 1 (identical).
    """

    texts: List[str]
    scores: List[float]

    @property
    def top_result(self) -> Optional[str]:
        """Returns the most relevant document text, or None if no results."""
        return self.texts[0] if self.texts else None
