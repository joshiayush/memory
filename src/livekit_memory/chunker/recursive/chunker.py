"""Recursive character text splitter chunking strategy."""

from __future__ import annotations

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


class RecursiveChunker:
    """Chunker using LangChain's RecursiveCharacterTextSplitter.

    This chunker splits text by recursively trying different separators
    (paragraphs, sentences, words) until chunks are small enough.
    Supports language-aware splitting for code files.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    # Map file extensions to Language enum
    LANGUAGE_MAP: dict[str, Language] = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".go": Language.GO,
        ".java": Language.JAVA,
        ".kt": Language.KOTLIN,
        ".rs": Language.RUST,
        ".rb": Language.RUBY,
        ".php": Language.PHP,
        ".swift": Language.SWIFT,
        ".scala": Language.SCALA,
        ".c": Language.C,
        ".cpp": Language.CPP,
        ".h": Language.C,
        ".hpp": Language.CPP,
        ".cs": Language.CSHARP,
        ".lua": Language.LUA,
        ".hs": Language.HASKELL,
        ".ex": Language.ELIXIR,
        ".exs": Language.ELIXIR,
        ".html": Language.HTML,
        ".htm": Language.HTML,
        ".sol": Language.SOL,
        ".cob": Language.COBOL,
        ".md": Language.MARKDOWN,
        ".markdown": Language.MARKDOWN,
        ".tex": Language.LATEX,
        ".rst": Language.RST,
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        language: Language | str | None = None,
    ) -> None:
        """Initialize the recursive chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separators to try, in order. Defaults to
                paragraphs, newlines, sentences, words. Ignored if language is set.
            language: Programming language for language-aware splitting.
                Can be a Language enum or file extension (e.g., ".py", ".js").
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators
        self._language = self._resolve_language(language)
        self._splitter = self._create_splitter()
        self._chunks: list[str] = []

    def _resolve_language(self, language: Language | str | None) -> Language | None:
        """Resolve language from enum or file extension."""
        if language is None:
            return None
        if isinstance(language, Language):
            return language
        # Treat as file extension
        ext = language if language.startswith(".") else f".{language}"
        return self.LANGUAGE_MAP.get(ext.lower())

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the appropriate splitter based on language setting."""
        if self._language is not None:
            return RecursiveCharacterTextSplitter.from_language(
                language=self._language,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators or self.DEFAULT_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def add_text(self, text: str) -> None:
        """Add text to be chunked.

        Args:
            text: The text content to chunk.
        """
        chunks = self._splitter.split_text(text)
        self._chunks.extend(chunks)

    def get_chunks(self) -> list[str]:
        """Get all chunks.

        Returns:
            List of text chunks.
        """
        return self._chunks

    def to_documents(self) -> list[str]:
        """Convert chunks to documents for vector database ingestion.

        Returns:
            List of document strings (same as get_chunks).
        """
        return self._chunks

    def clear(self) -> None:
        """Clear all stored chunks."""
        self._chunks = []
