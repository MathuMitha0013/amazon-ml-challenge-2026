"""
Business address normalization, token extraction, and numeric component isolation.
"""

import re
import unicodedata
from typing import Any

from unidecode import unidecode


# Common address abbreviations.
# These are generic and not country-specific.
ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln.": "ln",
    "court": "ct",
    "ct.": "ct",
    "parkway": "pkwy",
    "pkwy.": "pkwy",
    "highway": "hwy",
    "hwy.": "hwy",
    "place": "pl",
    "pl.": "pl",
    "suite": "ste",
    "ste.": "ste",
    "floor": "fl",
    "fl.": "fl",
    "apartment": "apt",
    "apt.": "apt",
    "building": "bldg",
    "bldg.": "bldg",
}


def normalize_business_address(raw_address: str) -> str:
    """
    Normalize a business address while preserving useful information.

    The normalization:
    - handles Unicode characters
    - transliterates non-Latin scripts
    - lowercases text
    - standardizes common address terms
    - preserves numbers
    - removes punctuation
    - normalizes whitespace
    """

    if raw_address is None:
        return ""

    address = str(raw_address).strip()

    if not address:
        return ""

    # Unicode normalization
    address = unicodedata.normalize("NFKD", address)

    # Transliteration
    address = unidecode(address)

    # Lowercase
    address = address.lower()

    # Normalize common separators
    address = address.replace("&", " and ")

    # Replace punctuation with spaces.
    # Numbers are intentionally preserved.
    address = re.sub(r"[^a-z0-9\s]", " ", address)

    # Normalize whitespace
    address = re.sub(r"\s+", " ", address).strip()

    # Standardize common address words
    tokens = []

    for token in address.split():
        tokens.append(
            ADDRESS_ABBREVIATIONS.get(token, token)
        )

    return " ".join(tokens)


def extract_address_components(raw_address: str) -> dict[str, Any]:
    """
    Extract useful structural components from an address.

    Returns:
        raw:
            Original address.

        normalized:
            Normalized address.

        numbers:
            Unique numeric components.

        postal_codes:
            Numeric components that look like postal/ZIP codes.

        street_numbers:
            Numeric components that appear as standalone
            building/street numbers.

        locality_tokens:
            Non-numeric address tokens.

        street_tokens:
            Tokens associated with common street/address terms.
    """

    normalized = normalize_business_address(raw_address)

    if not normalized:
        return {
            "raw": raw_address,
            "normalized": "",
            "numbers": [],
            "postal_codes": [],
            "street_numbers": [],
            "locality_tokens": [],
            "street_tokens": [],
        }

    # Extract numeric components.
    numbers = re.findall(
        r"\b\d+[a-z]?\b",
        normalized,
    )

    # Remove duplicate numbers while preserving order.
    numbers = list(dict.fromkeys(numbers))

    # Postal-code-like numeric components.
    # We intentionally don't hardcode a single country's format.
    postal_codes = [
        number
        for number in numbers
        if 4 <= len(re.sub(r"[^0-9]", "", number)) <= 6
    ]

    # Building/street numbers.
    street_numbers = []

    for match in re.finditer(
        r"(?:^|\s)(\d+[a-z]?)(?:\s|$)",
        normalized,
    ):
        street_numbers.append(match.group(1))

    street_numbers = list(
        dict.fromkeys(street_numbers)
    )

    # Tokenize address.
    tokens = normalized.split()

    # Common structural street terms.
    street_terms = {
        "st",
        "rd",
        "ave",
        "blvd",
        "dr",
        "ln",
        "ct",
        "pkwy",
        "hwy",
        "pl",
        "ste",
        "fl",
        "apt",
        "bldg",
    }

    street_tokens = [
        token
        for token in tokens
        if token in street_terms
    ]

    # Non-numeric tokens are useful for locality/address matching.
    locality_tokens = [
        token
        for token in tokens
        if not re.fullmatch(r"\d+[a-z]?", token)
    ]

    return {
        "raw": raw_address,
        "normalized": normalized,
        "numbers": numbers,
        "postal_codes": postal_codes,
        "street_numbers": list(
            dict.fromkeys(street_numbers)
        ),
        "locality_tokens": locality_tokens,
        "street_tokens": street_tokens,
    }
