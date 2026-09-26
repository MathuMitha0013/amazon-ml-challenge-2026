"""
Exact normalized-name blocking.

Generates candidate pairs when Source 1 and Source 2/3
have the same normalized business name and country.
"""

from typing import Any


def generate_exact_blocks(
    source1_data: Any,
    source2_3_data: Any,
    key_column: str = "name_normalized",
) -> Any:
    """
    Generate exact normalized-name candidate pairs.

    Required columns:
        Source 1:
            s1_id
            country
            name_normalized

        Source 2/3:
            matched_id
            country
            name_normalized

    Returns:
        DataFrame containing:
            s1_id
            matched_id
    """

    # Pandas DataFrame implementation
    candidates = source1_data.merge(
        source2_3_data,
        on=["country", key_column],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    return candidates[["s1_id", "matched_id"]].drop_duplicates()
