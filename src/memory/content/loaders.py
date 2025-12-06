"""Content loaders for different document types.

This module provides async functions to load content from various file formats
including plain text, markdown, and PDF documents.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Callable, Dict

from pypdf import PdfReader

from .types import DocType


_loader_registry: Dict[str, Callable] = dict()


def _register_loader(doc_type: DocType) -> Callable:
    """Decorator to register a loader function for a document type.

    Args:
        doc_type: The document type this loader handles.

    Returns:
        A decorator that registers the loader function and returns it unchanged.

    Example:
        >>> @_register_loader("text")
        ... async def load_text(path: Path) -> str:
        ...     return path.read_text()
    """

    def decorator(func: Callable) -> Callable:
        _loader_registry[doc_type] = func
        return func

    return decorator


@_register_loader("text")
async def load_text(path: Path) -> str:
    """Load content from a plain text file.

    Args:
        path: Path to the text file.

    Returns:
        The file content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


@_register_loader("markdown")
async def load_markdown(path: Path) -> str:
    """Load content from a markdown file.

    Args:
        path: Path to the markdown file.

    Returns:
        The file content as a string (raw markdown).

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


def _read_pdf_sync(path: Path) -> str:
    """Read PDF content synchronously (internal helper).

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text from all pages joined by newlines.
    """
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


@_register_loader("pdf")
async def load_pdf(path: Path) -> str:
    """Load content from a PDF file.

    Extracts text from all pages and joins them with newlines.

    Args:
        path: Path to the PDF file.

    Returns:
        The extracted text content.

    Raises:
        FileNotFoundError: If the file does not exist.
        pypdf.errors.PdfReadError: If the PDF is corrupted or invalid.
    """
    return await asyncio.to_thread(_read_pdf_sync, path)


async def load_document(path: Path, doc_type: DocType) -> str:
    """Load content from a document based on its type.

    Args:
        path: Path to the document.
        doc_type: The type of document to load.

    Returns:
        The document content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document type is not supported.
    """
    loader = _loader_registry.get(doc_type)
    if loader is None:
        raise ValueError(f"Unsupported document type: {doc_type}")

    return await loader(path)


def get_file_extension(doc_type: DocType) -> str:
    """Get the file extension for a document type.

    Args:
        doc_type: The document type.

    Returns:
        The file extension including the dot (e.g., ".txt").
    """
    extensions = {
        "text": ".txt",
        "markdown": ".md",
        "pdf": ".pdf",
    }
    return extensions[doc_type]


async def discover_files(
    path: Path,
    doc_type: DocType,
    recursive: bool = False,
) -> List[Path]:
    """Discover files of a given type in a path.

    Args:
        path: Path to a file or directory.
        doc_type: The type of documents to find.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of paths to matching files.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file or directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        return [path]

    if not path.is_dir():
        raise ValueError(f"Path is not a file or directory: {path}")

    ext = get_file_extension(doc_type)
    pattern = f"**/*{ext}" if recursive else f"*{ext}"

    return await asyncio.to_thread(lambda: list(path.glob(pattern)))
