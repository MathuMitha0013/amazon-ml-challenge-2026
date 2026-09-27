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

from unidecode import unidecode


LEGAL_SUFFIXES = [
    "private limited",
    "incorporated",
    "corporation",
    "company",
    "limited",
    "llc",
    "ltd",
    "inc",
    "corp",
    "co",
    "pc",
    "plc",
    "pllc",
]


def normalize_business_name(raw_name: str) -> str:
    """
    Normalize a business name while preserving useful discriminative tokens.

    The legal suffix is retained in the normalized representation.
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
    Generate multiple representations of a business name.

    Returns:
        raw:
            Original input.

        normalized:
            Cleaned normalized name with legal suffix retained.

        core:
            Normalized name with one trailing legal suffix removed.

        compact:
            Alphanumeric normalized representation without spaces.

        sorted_tokens:
            Alphabetically sorted normalized tokens.

        tokens:
            Individual normalized tokens.

        ngrams:
            Character 3-gram signatures.
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
