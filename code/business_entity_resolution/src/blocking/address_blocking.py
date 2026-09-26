"""
Address-based blocking route.

Generates candidate pairs using shared country and
discriminative address components such as numeric
identifiers and normalized address tokens.
"""

from typing import Any

import pandas as pd


def generate_address_blocks(
    source1_data: Any,
    source2_3_data: Any,
    min_token_length: int = 3,
) -> Any:
    """
    Generate candidate pairs using address components.

    Required columns:

    Source 1:
        s1_id
        country
        normalized_address

    Source 2/3:
        matched_id
        country
        normalized_address

    Returns:
        DataFrame containing:
            s1_id
            matched_id
    """

    s1 = source1_data[
        ["s1_id", "country", "normalized_address"]
    ].copy()

    target = source2_3_data[
        ["matched_id", "country", "normalized_address"]
    ].copy()

    # Handle missing addresses
    s1["normalized_address"] = (
        s1["normalized_address"]
        .fillna("")
        .astype(str)
    )

    target["normalized_address"] = (
        target["normalized_address"]
        .fillna("")
        .astype(str)
    )

    # Extract address tokens
    s1["address_token"] = (
        s1["normalized_address"].str.split()
    )
    target["address_token"] = (
        target["normalized_address"].str.split()
    )

    s1 = s1.explode("address_token")
    target = target.explode("address_token")

    # Remove short/generic tokens
    s1 = s1[
        s1["address_token"].str.len() >= min_token_length
    ]

    target = target[
        target["address_token"].str.len() >= min_token_length
    ]

    # Candidate generation using country + address token
    candidates = s1.merge(
        target,
        on=["country", "address_token"],
        how="inner",
        suffixes=("_s1", "_target"),
    )

    return candidates[
        ["s1_id", "matched_id"]
    ].drop_duplicates()
