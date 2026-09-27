"""
Character n-gram signature blocking route.

Responsibilities:
- Extract character 3-grams / 4-grams to capture spelling variations and typos in business names.
- Invert character n-grams within country with strict frequency thresholds.
- Retrieve candidates that share rare character n-gram signatures.
"""

from collections import defaultdict
import pandas as pd


def extract_ngrams_from_text(text: str, n: int = 3) -> set[str]:
    """Extracts unique character n-grams from words in the text."""
    if not text or len(text) < n:
        return set()
    res = set()
    for word in text.split():
        if len(word) >= n:
            for i in range(len(word) - n + 1):
                res.add(word[i : i + n])
    return res


def generate_ngram_blocks(
    source1_data: pd.DataFrame,
    source2_3_data: pd.DataFrame,
    ngram_size: int = 3,
    max_ngram_frequency: int = 50,
    key_column: str = "name_normalized",
) -> pd.DataFrame:
    """
    Generates candidate pairs based on shared rare character n-gram signatures within country.

    Args:
        source1_data: Master Source1 records (columns: s1_id, country, key_column).
        source2_3_data: Target noisy records (columns: matched_id, country, key_column).
        ngram_size: Size of character n-grams (default 3).
        max_ngram_frequency: Maximum frequency for an n-gram in target pool.
        key_column: Normalized name column.

    Returns:
        DataFrame of candidate pairs: ['s1_id', 'matched_id'].
    """
    if source1_data.empty or source2_3_data.empty or key_column not in source1_data.columns or key_column not in source2_3_data.columns:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    # Build target n-gram frequency and postings list
    ngram_counts = defaultdict(int)
    target_ngrams = defaultdict(list)

    for country, text_val, matched_id in zip(
        source2_3_data["country"].values,
        source2_3_data[key_column].values,
        source2_3_data["matched_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            ngs = extract_ngrams_from_text(str(text_val), n=ngram_size)
            for ng in ngs:
                key = (country, ng)
                ngram_counts[key] += 1
                target_ngrams[key].append(matched_id)

    # Query source1 n-grams against rare target postings
    pairs = set()
    for country, text_val, s1_id in zip(
        source1_data["country"].values,
        source1_data[key_column].values,
        source1_data["s1_id"].values,
    ):
        if text_val is not None and str(text_val) != "" and str(text_val) != "nan":
            ngs = extract_ngrams_from_text(str(text_val), n=ngram_size)
            for ng in ngs:
                key = (country, ng)
                if 0 < ngram_counts.get(key, 0) <= max_ngram_frequency:
                    for matched_id in target_ngrams[key]:
                        pairs.add((s1_id, matched_id))

    if not pairs:
        return pd.DataFrame(columns=["s1_id", "matched_id"])

    s1_ids, matched_ids = zip(*pairs)
    return pd.DataFrame({"s1_id": list(s1_ids), "matched_id": list(matched_ids)})
