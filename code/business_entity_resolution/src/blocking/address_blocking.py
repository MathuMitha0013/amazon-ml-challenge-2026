"""
Address-based blocking route.

Responsibilities:
- Extract address tokens (distinctive locality names, street tokens, postal codes, and numbers).
- Invert address tokens within country and filter out high-frequency street terms.
- Retrieve candidates that share rare address tokens or postal identifiers.
"""

from collections import defaultdict
import pandas as pd

import pandas as pd


def generate_address_blocks(
    source1_data: pd.DataFrame,
    source2_3_data: pd.DataFrame,
    max_token_frequency: int = 75,
    min_token_length: int = 3,
    key_column: str = "address_normalized",
) -> pd.DataFrame:
    """
    Blocks records based on shared numeric and locality address tokens within country boundaries.

    Args:
        source1_data: Master Source1 table (columns: s1_id, country, key_column).
        source2_3_data: Target noisy source tables (columns: matched_id, country, key_column).
        max_token_frequency: Maximum frequency for an address token in target pool.
        min_token_length: Minimum character length for address tokens.
        key_column: Normalized address column.

    Returns:
        DataFrame of candidate pairs: ['s1_id', 'matched_id'].
    """
    if source1_data.empty or source2_3_data.empty or key_column not in source1_data.columns or key_column not in source2_3_data.columns:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    # Build target address token frequency and postings list
    token_counts = defaultdict(int)
    target_tokens = defaultdict(list)

    for country, text_val, matched_id in zip(
        source2_3_data["country"].values,
        source2_3_data[key_column].values,
        source2_3_data["matched_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            tokens = set(str(text_val).split())
            for tok in tokens:
                if len(tok) >= min_token_length:
                    key = (country, tok)
                    token_counts[key] += 1
                    target_tokens[key].append(matched_id)

    # Query source1 address tokens against rare target postings
    pairs = set()
    for country, text_val, s1_id in zip(
        source1_data["country"].values,
        source1_data[key_column].values,
        source1_data["s1_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            tokens = set(str(text_val).split())
            for tok in tokens:
                if len(tok) >= min_token_length:
                    key = (country, tok)
                    if 0 < token_counts.get(key, 0) <= max_token_frequency:
                        for matched_id in target_tokens[key]:
                            pairs.add((s1_id, matched_id))

    if not pairs:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    s1_ids, matched_ids = zip(*pairs)
    return pd.DataFrame({"s1_id": list(s1_ids), "matched_id": list(matched_ids)})
