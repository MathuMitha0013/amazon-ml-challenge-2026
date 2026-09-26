"""
Business address normalization, token extraction, and numeric component isolation.

Future Responsibility:
- Standardize street, suite, floor, locality, and postal abbreviations.
- Isolate numeric components (PIN codes, zip codes, street numbers) for high-precision blocking.
- Extract geographic locality tokens without hardcoding specific country constraints.
- Retain raw address representation to prevent irreversible loss of granular detail.
"""

from typing import Any


def normalize_business_address(raw_address: str) -> str:
    """
    Cleans and standardizes an address string for text similarity evaluation.

    Args:
        raw_address: Raw address text.

    Returns:
        Cleaned, normalized address string.
    """
    raise NotImplementedError("normalize_business_address will be implemented in subsequent phases.")


def extract_address_components(raw_address: str) -> dict[str, Any]:
    """
    Decomposes an address into structural tokens:
    - numbers: set of extracted numeric strings (postal codes, building numbers)
    - locality_tokens: extracted geographic / city tokens
    - street_tokens: standardized street / directional words
    - raw: original address string

    Args:
        raw_address: Raw address string.

    Returns:
        Dictionary containing extracted components.
    """
    raise NotImplementedError("extract_address_components will be implemented in subsequent phases.")
