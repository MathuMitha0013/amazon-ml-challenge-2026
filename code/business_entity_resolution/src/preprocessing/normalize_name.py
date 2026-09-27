"""
Business name normalization and multi-representation generator.
"""

import re
import unicodedata
from typing import Any

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

    # Unicode normalization
    name = unicodedata.normalize("NFKD", name)

    # Transliterate non-Latin scripts
    name = unidecode(name)

    # Lowercase
    name = name.lower()

    # Handle dotted legal abbreviations before punctuation removal
    name = re.sub(
        r"\bl\s*\.?\s*l\s*\.?\s*c\.?\b",
        "llc",
        name,
    )

    name = re.sub(
        r"\bp\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\.?\b",
        "pllc",
        name,
    )

    name = re.sub(
        r"\bi\s*\.?\s*n\s*\.?\s*c\.?\b",
        "inc",
        name,
    )

    name = re.sub(
        r"\bl\s*\.?\s*t\s*\.?\s*d\.?\b",
        "ltd",
        name,
    )

    name = re.sub(
        r"\bp\s*\.?\s*c\.?\b",
        "pc",
        name,
    )

    # Normalize ampersand
    name = name.replace("&", " and ")

    # Remove punctuation/special characters
    name = re.sub(r"[^a-z0-9\s]", " ", name)

    # Normalize whitespace
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

    normalized = normalize_business_name(raw_name)

    # Remove one trailing legal suffix for the core representation
    core = normalized

    for suffix in sorted(
        LEGAL_SUFFIXES,
        key=len,
        reverse=True,
    ):
        pattern = rf"\b{re.escape(suffix)}$"

        if re.search(pattern, core):
            core = re.sub(
                pattern,
                "",
                core,
            ).strip()
            break

    tokens = core.split()

    # Compact representation
    compact = re.sub(
        r"[^a-z0-9]",
        "",
        normalized,
    )

    # Sorted token signature
    sorted_tokens = " ".join(
        sorted(tokens)
    )

    # Character 3-grams
    ngrams = []

    padded = f"  {core}  "

    if len(padded) >= 3:
        ngrams = [
            padded[i:i + 3]
            for i in range(len(padded) - 2)
        ]

    # Remove duplicate n-grams while preserving order
    ngrams = list(dict.fromkeys(ngrams))

    return {
        "raw": raw_name,
        "normalized": normalized,
        "core": core,
        "compact": compact,
        "sorted_tokens": sorted_tokens,
        "tokens": tokens,
        "ngrams": ngrams,
    }
