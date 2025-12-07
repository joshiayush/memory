"""Proposition extraction utilities.

This module provides functions to extract atomic propositions (sentences)
from text content for use with the AgenticChunker. Uses NLTK's Punkt
tokenizer for accurate sentence boundary detection.
"""

from __future__ import annotations

from typing import List

import nltk
from nltk.tokenize import PunktTokenizer

# Lazy-loaded tokenizer
_tokenizer: PunktTokenizer | None = None


def _get_tokenizer() -> PunktTokenizer:
    """Get or initialize the NLTK Punkt tokenizer.

    Downloads the required 'punkt_tab' data if not already present.

    Returns:
        The PunktTokenizer instance.
    """
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = PunktTokenizer()
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            _tokenizer = PunktTokenizer()
    return _tokenizer


def extract_propositions(text: str) -> List[str]:
    """Extract atomic propositions from text using NLP.

    Splits text into sentences using NLTK's Punkt sentence tokenizer,
    which handles abbreviations, titles, and other edge cases accurately.

    Args:
        text: The input text to split into propositions.

    Returns:
        List of sentences/propositions extracted from the text.
        Empty strings and whitespace-only sentences are filtered out.

    Example:
        >>> text = "Dr. Smith works at NASA. He studies Mars."
        >>> props = extract_propositions(text)
        >>> print(props)
        ['Dr. Smith works at NASA.', 'He studies Mars.']
    """
    if not text or not text.strip():
        return []

    tokenizer = _get_tokenizer()
    sentences = tokenizer.tokenize(text)

    # Filter empty strings and strip whitespace
    propositions = [s.strip() for s in sentences if s and s.strip()]

    return propositions
