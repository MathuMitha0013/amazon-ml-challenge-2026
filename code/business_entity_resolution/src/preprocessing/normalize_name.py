"""
Business name normalization and multi-representation generator.

Future Responsibility:
- Preserve raw names while constructing normalized, compact, and tokenized variants.
- Standardize legal suffixes (e.g., Ltd, Corp, LLC, Inc, Private Limited, GmbH, SA).
- Handle symbol substitutions ('&' vs 'and'), whitespace stripping, and sorted token signatures.
- Guard against over-normalization to avoid precision loss and false merges.
"""

from typing import Any


def normalize_business_name(raw_name: str) -> str:
    """
    Standardizes a business name string while retaining core discriminative tokens.

    Args:
        raw_name: Raw business name text.

    Returns:
        Normalized business name string.
    """
    raise NotImplementedError("normalize_business_name will be implemented in subsequent phases.")


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
    raise NotImplementedError("generate_name_representations will be implemented in subsequent phases.")
