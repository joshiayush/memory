"""Chunking strategies for document processing."""

from langchain_text_splitters import Language

from livekit_memory.chunker.recursive.chunker import RecursiveChunker

__all__ = ["AgenticChunker", "Language", "RecursiveChunker"]


def __getattr__(name: str):
    """Lazy import AgenticChunker to avoid loading heavy langchain dependencies (~180MB)."""
    if name == "AgenticChunker":
        from livekit_memory.chunker.agentic.chunker import AgenticChunker

        return AgenticChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
