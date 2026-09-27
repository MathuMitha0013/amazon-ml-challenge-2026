"""
Rare-token and inverted index blocking route.

Responsibilities:
- Build an inverted index over rare and discriminative business name tokens.
- Filter out ultra-frequent stopwords and generic terms (max_token_frequency).
- Retrieve candidate pairs sharing distinctive name tokens within country.
"""

from collections import defaultdict
import pandas as pd


def generate_token_blocks(
    source1_data: pd.DataFrame,
    source2_3_data: pd.DataFrame,
    max_token_frequency: int = 100,
    min_token_length: int = 3,
    key_column: str = "name_normalized",
) -> pd.DataFrame:
    """
    Generates candidate pairs using an inverted index of rare business name tokens within country.

    Args:
        source1_data: Master Source1 records (columns: s1_id, country, key_column).
        source2_3_data: Target records (columns: matched_id, country, key_column).
        max_token_frequency: Upper threshold for token frequency in target pool.
        min_token_length: Minimum character length for valid blocking tokens.
        key_column: Normalized name column.

    Returns:
        DataFrame of candidate pairs: ['s1_id', 'matched_id'].
    """
    if source1_data.empty or source2_3_data.empty or key_column not in source1_data.columns or key_column not in source2_3_data.columns:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    # Build target token frequency and postings list
    token_counts = defaultdict(int)
    target_tokens = defaultdict(list)

    for country, text_val, matched_id in zip(
        source2_3_data["country"].values,
        source2_3_data[key_column].values,
        source2_3_data["matched_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            tokens = str(text_val).split()
            for tok in tokens:
                if len(tok) >= min_token_length:
                    key = (country, tok)
                    token_counts[key] += 1
                    target_tokens[key].append(matched_id)

    # Query source1 tokens against rare target postings
    pairs = set()
    for country, text_val, s1_id in zip(
        source1_data["country"].values,
        source1_data[key_column].values,
        source1_data["s1_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            tokens = str(text_val).split()
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
