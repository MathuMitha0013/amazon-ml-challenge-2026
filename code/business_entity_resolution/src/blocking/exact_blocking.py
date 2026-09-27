"""
Exact normalized-name blocking.

Responsibilities:
- Generate high-precision candidate blocks where normalized business names match exactly within country.
- Provide deterministic O(1) hash table candidate lookups.
- Output candidate pairs in standardized schema: s1_id, matched_id.
"""

from collections import defaultdict
import pandas as pd


def generate_exact_blocks(
    source1_data: pd.DataFrame,
    source2_3_data: pd.DataFrame,
    key_column: str = "name_normalized",
) -> pd.DataFrame:
    """
    Performs exact key join between Source1 and Target (Source2/3) records within country boundaries.

    Args:
        source1_data: Master Source1 table (with columns s1_id, country, and key_column).
        source2_3_data: Target noisy source table (with columns matched_id, country, and key_column).
        key_column: Exact key column name to join on.

    Returns:
        DataFrame of candidate pairs: ['s1_id', 'matched_id'].
    """
    if source1_data.empty or source2_3_data.empty or key_column not in source1_data.columns or key_column not in source2_3_data.columns:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    # Build target index: (country, key) -> list[matched_id]
    target_index = defaultdict(list)
    for country, key_val, matched_id in zip(
        source2_3_data["country"].values,
        source2_3_data[key_column].values,
        source2_3_data["matched_id"].values,
    ):
        if key_val is not None and str(key_val) != "" and str(key_val) != "nan":
            target_index[(country, str(key_val))].append(matched_id)

    # Query source1 against target index
    pairs = set()
    for country, key_val, s1_id in zip(
        source1_data["country"].values,
        source1_data[key_column].values,
        source1_data["s1_id"].values,
    ):
        if key_val is not None and str(key_val) != "" and str(key_val) != "nan":
            query_key = (country, str(key_val))
            if query_key in target_index:
                for matched_id in target_index[query_key]:
                    pairs.add((s1_id, matched_id))

    if not pairs:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    s1_ids, matched_ids = zip(*pairs)
    return pd.DataFrame({"s1_id": list(s1_ids), "matched_id": list(matched_ids)})
