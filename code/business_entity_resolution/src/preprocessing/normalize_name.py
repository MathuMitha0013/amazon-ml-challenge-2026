"""
Business name normalization and multi-representation generator.

Responsibilities:
- Standardize legal suffixes (e.g., Ltd, Corp, LLC, Inc, PLLC, PC, etc.).
- Convert symbols ('&' to 'and').
- Strip punctuation, unidecode accents, convert to lowercase.
- Generate multiple representations (raw, normalized, compact, tokens, ngrams).
"""

import re
import unicodedata
from typing import Any
from unidecode import unidecode


def normalize_business_name(raw_name: str) -> str:
    """
    Standardizes a business name string while retaining core discriminative tokens.

    Args:
        raw_name: Raw business name text.

    Returns:
        Normalized business name string.
    """
    if raw_name is None:
        return ""
    name = str(raw_name).strip()
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = unidecode(name)
    name = name.lower()
    name = re.sub(r"\bl\s*\.?\s*l\s*\.?\s*c\.?\b", "llc", name)
    name = re.sub(r"\bp\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\.?\b", "pllc", name)
    name = re.sub(r"\bi\s*\.?\s*n\s*\.?\s*c\.?\b", "inc", name)
    name = re.sub(r"\bl\s*\.?\s*t\s*\.?\s*d\.?\b", "ltd", name)
    name = re.sub(r"\bp\s*\.?\s*c\.?\b", "pc", name)
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def generate_name_representations(raw_name: str) -> dict[str, Any]:
    """
    Generates multiple concurrent representations for downstream blocking and matching:
    - raw: original string
    - normalized: lowercased, cleaned legal suffixes
    - compact: alphanumeric only, no whitespace
    - sorted_tokens: alphabetically sorted distinctive tokens
    - ngrams: character 3-gram signatures

    Args:
        raw_name: Raw business name.

    Returns:
        Dictionary mapping representation keys to generated variants.
    """
    norm = normalize_business_name(raw_name)
    tokens = [t for t in norm.split() if t]
    sorted_tokens = " ".join(sorted(tokens))
    compact = re.sub(r"\s+", "", norm)

    ngrams = set()
    for t in tokens:
        if len(t) >= 3:
            for i in range(len(t) - 2):
                ngrams.add(t[i : i + 3])

    return {
        "raw": raw_name or "",
        "normalized": norm,
        "compact": compact,
        "sorted_tokens": sorted_tokens,
        "tokens": tokens,
        "ngrams": list(ngrams),
    }
