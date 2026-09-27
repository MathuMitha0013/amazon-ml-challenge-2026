"""
Business address normalization, token extraction, and numeric component isolation.

Responsibilities:
- Standardize street, suite, floor, locality, and postal abbreviations.
- Isolate numeric components (PIN codes, zip codes, building numbers).
- Extract locality and street tokens without hardcoding specific country constraints.
- Retain raw address representation to prevent loss of granular detail.
"""

import re
import unicodedata
from typing import Any
from unidecode import unidecode


def normalize_business_address(raw_address: str) -> str:
    """
    Cleans and standardizes an address string for text similarity evaluation and blocking.

    Args:
        raw_address: Raw address text.

    Returns:
        Cleaned, normalized address string.
    """
    if raw_address is None:
        return ""
    addr = str(raw_address).strip()
    if not addr:
        return ""
    addr = unicodedata.normalize("NFKD", addr)
    addr = unidecode(addr)
    addr = addr.lower()
    addr = re.sub(r"\brd\b", "road", addr)
    addr = re.sub(r"\bst\b", "street", addr)
    addr = re.sub(r"\bave\b", "avenue", addr)
    addr = re.sub(r"\bblvd\b", "boulevard", addr)
    addr = re.sub(r"\bflr\b", "floor", addr)
    addr = re.sub(r"\bste\b", "suite", addr)
    addr = re.sub(r"\bbldg\b", "building", addr)
    addr = re.sub(r"\bh\s*no\b", "hno", addr)
    addr = re.sub(r"\bplot\s*no\b", "plotno", addr)
    addr = re.sub(r"\bpo\s*box\b", "pobox", addr)
    addr = re.sub(r"[^a-z0-9\s]", " ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def extract_address_components(raw_address: str) -> dict[str, Any]:
    """
    Decomposes an address into structural tokens:
    - numbers: set of extracted numeric strings (postal codes, building numbers)
    - locality_tokens: extracted geographic / locality tokens
    - raw: original address string

    Args:
        raw_address: Raw address string.

    Returns:
        Dictionary containing extracted components.
    """
    norm = normalize_business_address(raw_address)
    tokens = norm.split()
    numbers = set(re.findall(r"\b\d+\b", norm))
    locality_tokens = set([t for t in tokens if len(t) >= 3 and not t.isdigit()])

    return {
        "raw": raw_address or "",
        "normalized": norm,
        "tokens": tokens,
        "numbers": numbers,
        "locality_tokens": locality_tokens,
    }
