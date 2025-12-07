"""Agentic chunker for semantic document segmentation.

This module provides an LLM-powered chunker that semantically groups
propositions (atomic statements) into coherent chunks based on meaning,
direction, and intent similarity.

Example:
    >>> from chunker.agentic import AgenticChunker
    >>> chunker = AgenticChunker()
    >>> chunker.add_propositions([
    ...     "Greg likes pizza.",
    ...     "Greg also enjoys burgers.",
    ...     "San Francisco has great weather.",
    ... ])
    >>> chunks = chunker.get_chunks()
"""

from __future__ import annotations

import uuid

from tidylog import get_logger
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

from .prompts import (
    FIND_RELEVANT_CHUNK_PROMPT,
    NEW_CHUNK_SUMMARY_PROMPT,
    NEW_CHUNK_TITLE_PROMPT,
    UPDATE_CHUNK_SUMMARY_PROMPT,
    UPDATE_CHUNK_TITLE_PROMPT,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a semantic chunk of related propositions.

    A chunk groups propositions that share similar meaning, direction, or
    intent. Each chunk has a title and summary that describe its contents
    at a generalized level.

    Attributes:
        chunk_id: Unique identifier for the chunk (truncated UUID).
        propositions: List of proposition strings belonging to this chunk.
        title: Brief descriptive title for the chunk.
        summary: One-sentence summary describing the chunk's contents.
        chunk_index: Zero-based index indicating creation order.
    """

    chunk_id: str
    propositions: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    chunk_index: int = 0


class ChunkIDResponse(BaseModel):
    """Structured response for chunk ID extraction.

    Used with LLM structured output to reliably extract chunk IDs from
    free-form LLM responses.

    Attributes:
        chunk_id: The matched chunk ID, or None if no match found.
    """

    chunk_id: str | None = Field(
        default=None,
        description="The chunk ID if a match was found, otherwise None",
    )


def create_default_llm() -> ChatVertexAI:
    """Create a default ChatVertexAI instance with Gemini 2.5 Flash.

    Returns:
        A ChatVertexAI instance configured with gemini-2.5-flash model
        and temperature set to 0 for deterministic outputs.
    """
    return ChatVertexAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


class AgenticChunker:
    """Semantically groups propositions into coherent chunks using an LLM.

    The AgenticChunker uses an LLM to determine which propositions belong
    together based on semantic similarity. It maintains chunk metadata
    (titles and summaries) that evolve as new propositions are added.

    Attributes:
        ID_TRUNCATE_LIMIT: Length of truncated UUIDs used as chunk IDs.

    Example:
        >>> chunker = AgenticChunker()
        >>> chunker.add_proposition("The author was born in 1990.")
        >>> chunker.add_proposition("The author grew up in Seattle.")
        >>> print(chunker.get_chunk_outline())
    """

    ID_TRUNCATE_LIMIT: ClassVar[int] = 5

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        generate_new_metadata: bool = True,
    ) -> None:
        """Initialize the AgenticChunker.

        Args:
            llm: Language model for semantic operations. If None, creates a
                default ChatVertexAI with gemini-2.5-flash.
            generate_new_metadata: Whether to update chunk summaries and
                titles when new propositions are added. Set to False to
                improve performance at the cost of metadata freshness.
        """
        self._chunks: dict[str, Chunk] = {}
        self._generate_new_metadata = generate_new_metadata
        self._llm = llm if llm is not None else create_default_llm()

    @property
    def chunks(self) -> dict[str, Chunk]:
        """Return the dictionary of chunks keyed by chunk ID."""
        return self._chunks

    def add_propositions(self, propositions: Iterable[str]) -> None:
        """Add multiple propositions to appropriate chunks.

        Each proposition is processed sequentially, either being added to
        an existing chunk with similar content or creating a new chunk.

        Args:
            propositions: Iterable of proposition strings to add.
        """
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition: str) -> None:
        """Add a single proposition to the appropriate chunk.

        Finds a semantically similar existing chunk or creates a new one.
        If generate_new_metadata is enabled, updates the chunk's summary
        and title after adding the proposition.

        Args:
            proposition: The proposition string to add.
        """
        logger.info("adding proposition", extra={"proposition": proposition})

        # First proposition always creates a new chunk
        if not self._chunks:
            logger.info("no chunks exist, creating a new one")
            self._create_new_chunk(proposition)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        if chunk_id:
            chunk = self._chunks[chunk_id]
            logger.info(
                "chunk found, adding proposition",
                extra={"chunk_id": chunk.chunk_id, "title": chunk.title},
            )
            self._add_proposition_to_chunk(chunk_id, proposition)
        else:
            logger.info("no matching chunk found, creating a new one")
            self._create_new_chunk(proposition)

    def _add_proposition_to_chunk(
        self,
        chunk_id: str,
        proposition: str,
    ) -> None:
        """Add a proposition to an existing chunk and update metadata.

        Args:
            chunk_id: ID of the chunk to add the proposition to.
            proposition: The proposition string to add.
        """
        chunk = self._chunks[chunk_id]
        chunk.propositions.append(proposition)

        if self._generate_new_metadata:
            chunk.summary = self._update_chunk_summary(chunk)
            chunk.title = self._update_chunk_title(chunk)

    def _invoke_llm(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        """Invoke the LLM with a prompt and return the response content.

        Args:
            prompt: The ChatPromptTemplate to use.
            **kwargs: Variables to fill in the prompt template.

        Returns:
            The string content of the LLM's response.
        """
        runnable = prompt | self._llm | StrOutputParser()
        response = runnable.invoke(kwargs)
        return response

    def _update_chunk_summary(self, chunk: Chunk) -> str:
        """Generate an updated summary for a chunk.

        Called after adding a new proposition to keep the summary current
        and reflective of all propositions in the chunk.

        Args:
            chunk: The chunk to generate an updated summary for.

        Returns:
            The new summary string.
        """
        return self._invoke_llm(
            UPDATE_CHUNK_SUMMARY_PROMPT,
            propositions="\n".join(chunk.propositions),
            current_summary=chunk.summary,
        )

    def _update_chunk_title(self, chunk: Chunk) -> str:
        """Generate an updated title for a chunk.

        Called after adding a new proposition to keep the title current
        and reflective of all propositions in the chunk.

        Args:
            chunk: The chunk to generate an updated title for.

        Returns:
            The new title string.
        """
        return self._invoke_llm(
            UPDATE_CHUNK_TITLE_PROMPT,
            propositions="\n".join(chunk.propositions),
            current_summary=chunk.summary,
            current_title=chunk.title,
        )

    def _get_new_chunk_summary(self, proposition: str) -> str:
        """Generate an initial summary for a new chunk.

        Args:
            proposition: The first proposition in the new chunk.

        Returns:
            The initial summary string.
        """
        return self._invoke_llm(
            NEW_CHUNK_SUMMARY_PROMPT,
            proposition=proposition,
        )

    def _get_new_chunk_title(self, summary: str) -> str:
        """Generate an initial title for a new chunk.

        Args:
            summary: The summary of the new chunk.

        Returns:
            The initial title string.
        """
        return self._invoke_llm(
            NEW_CHUNK_TITLE_PROMPT,
            summary=summary,
        )

    def _create_new_chunk(self, proposition: str) -> None:
        """Create a new chunk with the given proposition.

        Generates a unique ID, summary, and title for the new chunk.

        Args:
            proposition: The first proposition for the new chunk.
        """
        chunk_id = str(uuid.uuid4())[: self.ID_TRUNCATE_LIMIT]
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)

        self._chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            propositions=[proposition],
            title=title,
            summary=summary,
            chunk_index=len(self._chunks),
        )

        logger.info(
            "created new chunk",
            extra={"chunk_id": chunk_id, "title": title},
        )

    def get_chunk_outline(self) -> str:
        """Get a formatted string representation of all chunks.

        Useful for displaying the current state of chunks or for providing
        context to the LLM when finding relevant chunks.

        Returns:
            A formatted string with each chunk's ID, title, and summary.
            Returns an empty string if no chunks exist.
        """
        lines = []
        for chunk in self._chunks.values():
            lines.append(
                f"Chunk ({chunk.chunk_id}): {chunk.title}\n"
                f"Summary: {chunk.summary}\n"
            )
        return "\n".join(lines)

    def _find_relevant_chunk(self, proposition: str) -> str | None:
        """Find an existing chunk that matches the proposition.

        Uses the LLM to determine semantic similarity between the
        proposition and existing chunks.

        Args:
            proposition: The proposition to find a matching chunk for.

        Returns:
            The chunk ID if a match is found, None otherwise.
        """
        current_outline = self.get_chunk_outline()

        # Get initial LLM response
        raw_response = self._invoke_llm(
            FIND_RELEVANT_CHUNK_PROMPT,
            current_chunk_outline=current_outline,
            proposition=proposition,
        )

        # Use structured output to extract chunk ID reliably
        structured_llm = self._llm.with_structured_output(ChunkIDResponse)
        extraction_prompt = (
            f"Extract the chunk ID from this response. If no chunk was found "
            f"or the response indicates 'No chunks', return null.\n\n"
            f"Response: {raw_response}"
        )
        result: ChunkIDResponse = structured_llm.invoke(extraction_prompt)

        # Validate the extracted chunk ID
        if result.chunk_id and len(result.chunk_id) == self.ID_TRUNCATE_LIMIT:
            return result.chunk_id

        return None

    def get_chunks(
        self,
        output_format: Literal["dict", "list_of_strings"] = "dict",
    ) -> dict[str, Chunk] | list[str]:
        """Return chunks in the specified format.

        Args:
            output_format: The output format. "dict" returns the internal
                chunk dictionary. "list_of_strings" returns a list where
                each element is all propositions in a chunk joined by spaces.

        Returns:
            Either a dictionary of Chunk objects keyed by chunk ID, or a
            list of strings where each string contains all propositions
            from a single chunk.

        Raises:
            ValueError: If output_format is not "dict" or "list_of_strings".
        """
        if output_format == "dict":
            return self._chunks
        if output_format == "list_of_strings":
            return self.to_documents()
        msg = f"Invalid output_format: {output_format}"
        raise ValueError(msg)

    def to_documents(self) -> list[str]:
        """Convert chunks to document strings for vector database ingestion.

        Each chunk becomes a single document string with all its propositions
        joined by spaces. This format is compatible with Qdrant's
        `ingest_documents` method which expects `List[str]`.

        Returns:
            List of document strings, one per chunk, ready for embedding
            and storage in a vector database.

        Example:
            >>> chunker = AgenticChunker()
            >>> chunker.add_propositions(["Fact 1.", "Fact 2."])
            >>> documents = chunker.to_documents()
            >>> await pipeline.ingest_documents(documents)
        """
        return [" ".join(chunk.propositions) for chunk in self._chunks.values()]
