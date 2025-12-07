# livekit-memory

Document ingestion engine. Chunks and indexes knowledge sources into Qdrant to power LiveKit agent's contextual retrieval.

## Installation

```bash
pip install livekit-memory
```

## Usage

### CLI

**Ingest documents:**

```bash
# Ingest a markdown file to Qdrant Cloud
livekit-memory ingest document.md --type markdown --collection my-docs \
    --url https://your-cluster.qdrant.io --api-key $QDRANT_API_KEY

# Ingest to localhost Qdrant (no API key required)
livekit-memory ingest --url localhost --collection my-docs --file document.md --type markdown
```

**Query documents:**

```bash
# Query from Qdrant Cloud
livekit-memory query --url https://your-cluster.qdrant.io --api-key $QDRANT_API_KEY \
    --collection my-docs --query "What is the main topic?"

# Query from localhost
livekit-memory query --url localhost --collection my-docs --query "What is the main topic?"
```

**Migrate between Qdrant instances:**

```bash
# Migrate from cloud to localhost
livekit-memory migrate \
    --source-url https://your-cluster.qdrant.io --source-api-key $QDRANT_API_KEY \
    --dest-url localhost \
    --collection my-docs
```

### Python API

```python
import asyncio
from livekit_memory import FastRAGPipeline, RAGConfig, QdrantConfig, EmbeddingConfig

async def main():
    config = RAGConfig(
        qdrant=QdrantConfig(
            url="localhost",
            collection_name="my-docs",
        ),
        embedding=EmbeddingConfig(
            model_name="BAAI/bge-small-en-v1.5",
        ),
    )

    pipeline = FastRAGPipeline(config)
    await pipeline.initialize_collection()

    # Ingest documents
    await pipeline.ingest_documents(["Document content here..."])

    # Retrieve context
    result = await pipeline.retrieve_context("What is this about?", top_k=5)
    print(result.top_result)

asyncio.run(main())
```

## License

Apache-2.0
