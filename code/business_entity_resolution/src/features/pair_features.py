"""
Pairwise similarity feature engineering for Entity Matching.

Calculates comprehensive name, address, structural, and metadata similarity features
for candidate pairs using RapidFuzz, set operations, and vectorized computations.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence
import polars as pl
import numpy as np
from rapidfuzz import fuzz, distance


def _clean_str(val: Any) -> str:
    """Safely convert value to stripped string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _normalize_text_simple(text: str) -> str:
    """Lowercase and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _extract_tokens(text: str) -> list[str]:
    """Extract list of alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _extract_digits(text: str) -> set[str]:
    """Extract set of standalone digit tokens."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


FEATURE_COLUMNS = [
    # Name features
    "name_exact_match",
    "name_norm_exact_match",
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_wratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_len_diff",
    "name_len_ratio",
    # Address features
    "addr_exact_match",
    "addr_norm_exact_match",
    "addr_fuzz_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_num_overlap",
    "addr_num_exact_agreement",
    "addr_locality_overlap",
    "addr_len_diff",
    "addr_len_ratio",
    # Context & Metadata
    "country_match",
    "country_missing",
    "is_s2",
    "is_s3",
    "num_blocks_hit",
]


def extract_single_pair_features(
    s1_name: str,
    c_name: str,
    s1_addr: str,
    c_addr: str,
    s1_country: str,
    c_country: str,
    candidate_id: str = "",
    num_blocks_hit: int = 1,
) -> dict[str, float]:
    """
    Extracts complete pairwise similarity feature dictionary for a single pair.

    Args:
        s1_name: Source1 business name.
        c_name: Candidate business name.
        s1_addr: Source1 address.
        c_addr: Candidate address.
        s1_country: Source1 country.
        c_country: Candidate country.
        candidate_id: Candidate entity ID (e.g. S2-..., S3-...).
        num_blocks_hit: Number of blocking routes that proposed this candidate.

    Returns:
        Dictionary mapping feature name to float value.
    """
    s1_name_raw = _clean_str(s1_name)
    c_name_raw = _clean_str(c_name)
    s1_addr_raw = _clean_str(s1_addr)
    c_addr_raw = _clean_str(c_addr)
    s1_c_raw = _clean_str(s1_country)
    c_c_raw = _clean_str(c_country)

    s1_name_norm = _normalize_text_simple(s1_name_raw)
    c_name_norm = _normalize_text_simple(c_name_raw)
    s1_addr_norm = _normalize_text_simple(s1_addr_raw)
    c_addr_norm = _normalize_text_simple(c_addr_raw)

    # Token sets
    s1_n_toks = set(_extract_tokens(s1_name_norm))
    c_n_toks = set(_extract_tokens(c_name_norm))

    s1_a_toks = set(_extract_tokens(s1_addr_norm))
    c_a_toks = set(_extract_tokens(c_addr_norm))

    # Name features
    name_exact = 1.0 if s1_name_raw and (s1_name_raw.lower() == c_name_raw.lower()) else 0.0
    name_norm_exact = 1.0 if s1_name_norm and (s1_name_norm == c_name_norm) else 0.0

    if s1_name_norm and c_name_norm:
        name_fuzz_ratio = fuzz.ratio(s1_name_norm, c_name_norm) / 100.0
        name_partial = fuzz.partial_ratio(s1_name_norm, c_name_norm) / 100.0
        name_wratio = fuzz.WRatio(s1_name_norm, c_name_norm) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name_norm, c_name_norm) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_name_norm, c_name_norm) / 100.0
    else:
        name_fuzz_ratio = name_partial = name_wratio = name_token_sort = name_token_set = 0.0

    # Name token overlap & Jaccard
    n_inter = len(s1_n_toks & c_n_toks)
    n_union = len(s1_n_toks | c_n_toks)
    name_token_jaccard = float(n_inter / n_union) if n_union > 0 else 0.0
    name_token_overlap = float(n_inter / min(len(s1_n_toks), len(c_n_toks))) if s1_n_toks and c_n_toks else 0.0

    # Name length metrics
    l1, l2 = len(s1_name_norm), len(c_name_norm)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    # Address features
    addr_exact = 1.0 if s1_addr_raw and (s1_addr_raw.lower() == c_addr_raw.lower()) else 0.0
    addr_norm_exact = 1.0 if s1_addr_norm and (s1_addr_norm == c_addr_norm) else 0.0

    if s1_addr_norm and c_addr_norm:
        addr_fuzz_ratio = fuzz.ratio(s1_addr_norm, c_addr_norm) / 100.0
        addr_partial = fuzz.partial_ratio(s1_addr_norm, c_addr_norm) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr_norm, c_addr_norm) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_addr_norm, c_addr_norm) / 100.0
    else:
        addr_fuzz_ratio = addr_partial = addr_token_sort = addr_token_set = 0.0

    # Address token overlap & Jaccard
    a_inter = len(s1_a_toks & c_a_toks)
    a_union = len(s1_a_toks | c_a_toks)
    addr_token_jaccard = float(a_inter / a_union) if a_union > 0 else 0.0
    addr_token_overlap = float(a_inter / min(len(s1_a_toks), len(c_a_toks))) if s1_a_toks and c_a_toks else 0.0

    # Numeric address features (postal codes, street numbers)
    s1_nums = _extract_digits(s1_addr_raw)
    c_nums = _extract_digits(c_addr_raw)
    if s1_nums and c_nums:
        num_inter = len(s1_nums & c_nums)
        num_union = len(s1_nums | c_nums)
        addr_num_overlap = float(num_inter / num_union)
        addr_num_exact = 1.0 if s1_nums == c_nums else 0.0
    elif not s1_nums and not c_nums:
        addr_num_overlap = 0.5
        addr_num_exact = 0.5
    else:
        addr_num_overlap = 0.0
        addr_num_exact = 0.0

    # Locality overlap: non-numeric tokens overlap in address
    s1_loc = s1_a_toks - s1_nums
    c_loc = c_a_toks - c_nums
    loc_inter = len(s1_loc & c_loc)
    loc_union = len(s1_loc | c_loc)
    addr_locality_overlap = float(loc_inter / loc_union) if loc_union > 0 else 0.0

    # Address length metrics
    al1, al2 = len(s1_addr_norm), len(c_addr_norm)
    addr_len_diff = float(abs(al1 - al2))
    addr_len_ratio = float(min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 0.0

    # Country features
    if not s1_c_raw or not c_c_raw:
        country_match = 0.5
        country_missing = 1.0
    elif s1_c_raw.upper() == c_c_raw.upper():
        country_match = 1.0
        country_missing = 0.0
    else:
        country_match = 0.0
        country_missing = 0.0

    # Source indicators
    cid = candidate_id.upper()
    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    is_s3 = 1.0 if cid.startswith("S3-") else 0.0

    return {
        "name_exact_match": name_exact,
        "name_norm_exact_match": name_norm_exact,
        "name_fuzz_ratio": name_fuzz_ratio,
        "name_partial_ratio": name_partial,
        "name_wratio": name_wratio,
        "name_token_sort_ratio": name_token_sort,
        "name_token_set_ratio": name_token_set,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_exact_match": addr_exact,
        "addr_norm_exact_match": addr_norm_exact,
        "addr_fuzz_ratio": addr_fuzz_ratio,
        "addr_partial_ratio": addr_partial,
        "addr_token_sort_ratio": addr_token_sort,
        "addr_token_set_ratio": addr_token_set,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_token_overlap": addr_token_overlap,
        "addr_num_overlap": addr_num_overlap,
        "addr_num_exact_agreement": addr_num_exact,
        "addr_locality_overlap": addr_locality_overlap,
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "country_match": country_match,
        "country_missing": country_missing,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "num_blocks_hit": float(num_blocks_hit),
    }


def extract_pair_features(
    df: pl.DataFrame,
    s1_name_col: str = "s1_business_name",
    candidate_name_col: str = "candidate_business_name",
    s1_addr_col: str = "s1_business_address",
    candidate_addr_col: str = "candidate_business_address",
    s1_country_col: str = "s1_country",
    candidate_country_col: str = "candidate_country",
    candidate_id_col: str = "candidate_entity_id",
    num_blocks_hit_col: Optional[str] = "number_of_blocks_hit",
) -> pl.DataFrame:
    """
    Vectorized extraction of pairwise similarity feature matrix from candidate pairs DataFrame.

    Args:
        df: Polars DataFrame of candidate pairs with text attributes.
        s1_name_col: S1 business name column.
        candidate_name_col: Candidate business name column.
        s1_addr_col: S1 address column.
        candidate_addr_col: Candidate address column.
        s1_country_col: S1 country column.
        candidate_country_col: Candidate country column.
        candidate_id_col: Candidate entity ID column.
        num_blocks_hit_col: Number of blocks hit column (if present).

    Returns:
        Polars DataFrame augmented with all engineered FEATURE_COLUMNS.
    """
    if df.height == 0:
        empty_schema = {col: pl.Float64 for col in FEATURE_COLUMNS}
        return df.with_columns([pl.lit(0.0).alias(c) for c in FEATURE_COLUMNS])

    s1_names = df[s1_name_col].fill_null("").to_list() if s1_name_col in df.columns else [""] * df.height
    c_names = df[candidate_name_col].fill_null("").to_list() if candidate_name_col in df.columns else [""] * df.height

    s1_addrs = df[s1_addr_col].fill_null("").to_list() if s1_addr_col in df.columns else [""] * df.height
    c_addrs = df[candidate_addr_col].fill_null("").to_list() if candidate_addr_col in df.columns else [""] * df.height

    s1_countries = df[s1_country_col].fill_null("").to_list() if s1_country_col in df.columns else [""] * df.height
    c_countries = df[candidate_country_col].fill_null("").to_list() if candidate_country_col in df.columns else [""] * df.height

    c_ids = df[candidate_id_col].fill_null("").to_list() if candidate_id_col in df.columns else [""] * df.height

    has_hit = num_blocks_hit_col and num_blocks_hit_col in df.columns
    hit_counts = df[num_blocks_hit_col].fill_null(1).to_list() if has_hit else [1] * df.height

    n = df.height
    feat_arrays = {col: np.zeros(n, dtype=np.float32) for col in FEATURE_COLUMNS}

    for i in range(n):
        feats = extract_single_pair_features(
            s1_name=s1_names[i],
            c_name=c_names[i],
            s1_addr=s1_addrs[i],
            c_addr=c_addrs[i],
            s1_country=s1_countries[i],
            c_country=c_countries[i],
            candidate_id=c_ids[i],
            num_blocks_hit=hit_counts[i],
        )
        for col, val in feats.items():
            feat_arrays[col][i] = val

    feature_series = [pl.Series(col, feat_arrays[col]) for col in FEATURE_COLUMNS]
    return df.with_columns(feature_series)


def standardize_candidate_schema(df: Any) -> pl.DataFrame:
    """
    Standardizes candidate pairs schema from any Person 2 format to canonical schema.
    Converts s1_id -> source1_entity_id, matched_id/matched_entity_id -> candidate_entity_id.
    """
    if not isinstance(df, pl.DataFrame):
        if hasattr(df, "to_pandas"):
            df = pl.from_pandas(df.to_pandas())
        else:
            df = pl.from_pandas(df)

    rename_map = {}
    if "s1_id" in df.columns and "source1_entity_id" not in df.columns:
        rename_map["s1_id"] = "source1_entity_id"
    if "matched_id" in df.columns and "candidate_entity_id" not in df.columns:
        rename_map["matched_id"] = "candidate_entity_id"
    elif "matched_entity_id" in df.columns and "candidate_entity_id" not in df.columns:
        rename_map["matched_entity_id"] = "candidate_entity_id"

    if rename_map:
        df = df.rename(rename_map)

    # Ensure required ID columns exist
    if "source1_entity_id" not in df.columns or "candidate_entity_id" not in df.columns:
        raise ValueError(f"Candidate DataFrame must have Source1 and Candidate ID columns. Found: {df.columns}")

    return df


def hydrate_candidate_pairs(
    candidate_pairs: Any,
    source1_table: Any,
    source2_table: Optional[Any] = None,
    source3_table: Optional[Any] = None,
    target_table: Optional[Any] = None,
) -> pl.DataFrame:
    """
    Hydrates candidate pairs by joining Source 1, Source 2, and Source 3 text attributes on entity IDs.
    Operates via out-of-core / lazy Polars joins without creating huge in-memory cartesian products.

    Args:
        candidate_pairs: DataFrame or relation with candidate ID pairs.
        source1_table: Master Source1 table or DataFrame (entity_id, business_name, business_address, country).
        source2_table: Source2 table.
        source3_table: Source3 table.
        target_table: Combined S2/S3 target table (alternative to separate source2/source3).

    Returns:
        Polars DataFrame with hydrated text attributes and candidate metadata.
    """
    cand_df = standardize_candidate_schema(candidate_pairs)

    def _to_polars_lazy(t: Any) -> pl.LazyFrame:
        if isinstance(t, pl.LazyFrame):
            return t
        elif isinstance(t, pl.DataFrame):
            return t.lazy()
        elif hasattr(t, "to_pandas"):
            return pl.from_pandas(t.to_pandas()).lazy()
        elif isinstance(t, (str, Path)):
            p = str(t)
            if p.endswith(".parquet"):
                return pl.scan_parquet(p)
            else:
                return pl.scan_csv(p, separator="\t")
        else:
            return pl.from_pandas(t).lazy()

    s1_lazy = _to_polars_lazy(source1_table).select(
        [
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("business_name").alias("s1_business_name"),
            pl.col("business_address").alias("s1_business_address"),
            pl.col("country").alias("s1_country"),
        ]
    )

    # Prepare combined target table
    if target_table is not None:
        target_lazy = _to_polars_lazy(target_table)
    elif source2_table is not None and source3_table is not None:
        s2_lazy = _to_polars_lazy(source2_table)
        s3_lazy = _to_polars_lazy(source3_table)
        target_lazy = pl.concat([s2_lazy, s3_lazy])
    elif source2_table is not None:
        target_lazy = _to_polars_lazy(source2_table)
    else:
        raise ValueError("Must provide either target_table or source2_table + source3_table")

    target_prep = target_lazy.select(
        [
            pl.col("entity_id").alias("candidate_entity_id"),
            pl.col("business_name").alias("candidate_business_name"),
            pl.col("business_address").alias("candidate_business_address"),
            pl.col("country").alias("candidate_country"),
        ]
    )

    # Perform lazy left joins
    hydrated_lazy = (
        cand_df.lazy()
        .join(s1_lazy, on="source1_entity_id", how="left")
        .join(target_prep, on="candidate_entity_id", how="left")
    )

    return hydrated_lazy.collect()


def construct_training_candidates(
    candidate_pairs: Any,
    ground_truth: Any,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
) -> pl.DataFrame:
    """
    Labels candidate pairs using ground truth:
    - label = 1 if (s1_id, candidate_id) is in ground truth
    - label = 0 for all other candidate pairs (hard negatives generated by blocking).

    Args:
        candidate_pairs: DataFrame of candidate pairs.
        ground_truth: Ground truth mapping, DataFrame, or TSV path.
        s1_id_col: S1 ID column.
        candidate_id_col: Candidate ID column.

    Returns:
        Polars DataFrame augmented with binary 'label' column (int32).
    """
    cand_df = standardize_candidate_schema(candidate_pairs)

    # Build ground truth positive pairs set
    gt_pairs_set = set()
    if isinstance(ground_truth, dict):
        for s1, c_ids in ground_truth.items():
            for c in c_ids:
                gt_pairs_set.add((str(s1).strip(), str(c).strip()))
    else:
        if isinstance(ground_truth, (str, Path)):
            gt_df = pl.read_csv(ground_truth, separator="\t")
        elif hasattr(ground_truth, "to_pandas"):
            gt_df = pl.from_pandas(ground_truth.to_pandas())
        else:
            gt_df = ground_truth

        for row in gt_df.iter_rows(named=True):
            s1 = str(row["source1_entity_id"]).strip()
            matched_str = row.get("matched_entity_ids")
            if matched_str and str(matched_str).strip() and str(matched_str).lower() != "null":
                for cid in str(matched_str).split(","):
                    cid_clean = cid.strip()
                    if cid_clean:
                        gt_pairs_set.add((s1, cid_clean))

    # Fast vectorised label assignment
    s1_arr = cand_df[s1_id_col].to_numpy()
    c_arr = cand_df[candidate_id_col].to_numpy()
    n = len(s1_arr)
    labels = np.zeros(n, dtype=np.int32)

    for i in range(n):
        if (s1_arr[i], c_arr[i]) in gt_pairs_set:
            labels[i] = 1

    return cand_df.with_columns(pl.Series("label", labels))


def extract_batch_features(
    candidate_pairs_table: Any,
    source1_table: Optional[Any] = None,
    source2_3_table: Optional[Any] = None,
    batch_size: int = 100_000,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Extracts features in memory-friendly batches with automatic schema standardization.

    Args:
        candidate_pairs_table: DataFrame of candidate pairs.
        source1_table: Optional Source1 master table to join on ID if text columns missing.
        source2_3_table: Optional target noisy source table to join on ID if text columns missing.
        batch_size: Number of rows per batch.

    Returns:
        DataFrame with feature columns.
    """
    df = standardize_candidate_schema(candidate_pairs_table)

    if source1_table is not None and "s1_business_name" not in df.columns:
        df = hydrate_candidate_pairs(df, source1_table=source1_table, target_table=source2_3_table)

    if df.height <= batch_size:
        return extract_pair_features(df, **kwargs)

    chunks = []
    for offset in range(0, df.height, batch_size):
        chunk = df.slice(offset, batch_size)
        feat_chunk = extract_pair_features(chunk, **kwargs)
        chunks.append(feat_chunk)

    return pl.concat(chunks)
