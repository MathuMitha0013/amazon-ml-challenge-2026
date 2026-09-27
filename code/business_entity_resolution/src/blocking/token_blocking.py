"""
Rare-token inverted-index blocking.

Generates candidate pairs when Source 1 and Source 2/3 share
a distinctive business-name token within the same country.
"""

from typing import Any
import pandas as pd


def generate_token_blocks(
    source1_data: Any,
    source2_3_data: Any,
    max_token_frequency: int = 100,
    min_token_length: int = 3,
) -> Any:
    """
    Generate candidate pairs using rare business-name tokens.

    Required columns:
        Source 1:
            s1_id
            country
            name_normalized

        Source 2/3:
            matched_id
            country
            name_normalized

    Args:
        source1_data: Source 1 DataFrame.
        source2_3_data: Source 2/3 DataFrame.
        max_token_frequency: Maximum token frequency allowed.
        min_token_length: Minimum token length.

    Returns:
        DataFrame containing:
            s1_id
            matched_id
    """

    s1 = source1_data[
        ["s1_id", "country", "name_normalized"]
    ].copy()

    target = source2_3_data[
        ["matched_id", "country", "name_normalized"]
    ].copy()

    # Create token rows
    s1["token"] = s1["name_normalized"].fillna("").str.split()
    s1 = s1.explode("token")

    target["token"] = target["name_normalized"].fillna("").str.split()
    target = target.explode("token")

    # Remove short tokens
    s1 = s1[s1["token"].str.len() >= min_token_length]
    target = target[target["token"].str.len() >= min_token_length]

    # Calculate token frequency in target data by country
    token_frequency = (
        target.groupby(["country", "token"])
        .size()
        .reset_index(name="frequency")
    )

    # Keep only rare/discriminative tokens
    token_frequency = token_frequency[
        token_frequency["frequency"] <= max_token_frequency
    ]

    # Keep only valid tokens in both datasets
    s1 = s1.merge(
        token_frequency,
        on=["country", "token"],
        how="inner",
    )

    target = target.merge(
        token_frequency,
        on=["country", "token"],
        how="inner",
    )

    # Inverted-index style join
    candidates = s1.merge(
        target,
        on=["country", "token"],
        how="inner",
        suffixes=("_s1", "_target"),
    )

    return candidates[
        ["s1_id", "matched_id"]
    ].drop_duplicates()
