"""
Batch inference, prediction scoring, threshold gating, and submission formatting.

Loads trained EntityMatcherModel artifact and optimal threshold config,
applies decision rule (score >= threshold -> MATCH, score < threshold -> NON-MATCH),
prints evaluation & match metrics to console, and exports format-compliant predictions to output/matching_results.tsv.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import polars as pl
import pandas as pd
import numpy as np

# Dynamically add project root folder to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.models.train_model import EntityMatcherModel


def predict_matches_batch(
    model: EntityMatcherModel,
    candidate_df: pl.DataFrame,
    threshold: float = 0.50,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
    batch_size: int = 200_000,
) -> pl.DataFrame:
    """
    Computes match probabilities for candidate pairs and applies the decision threshold:
    - score >= threshold -> MATCH (accepted match pair)
    - score < threshold  -> NON-MATCH (filtered out)

    Args:
        model: Fitted EntityMatcherModel.
        candidate_df: Candidate pairs DataFrame with extracted features.
        threshold: Optimal decision threshold cutoff probability.
        s1_id_col: Source1 ID column name.
        candidate_id_col: Candidate ID column name.
        batch_size: Chunk size for scoring.

    Returns:
        Polars DataFrame of accepted MATCH pairs with columns [s1_id_col, candidate_id_col, 'match_probability'].
    """
    if candidate_df.height == 0:
        return pl.DataFrame(
            {
                s1_id_col: pl.Series([], dtype=pl.Utf8),
                candidate_id_col: pl.Series([], dtype=pl.Utf8),
                "match_probability": pl.Series([], dtype=pl.Float32),
            }
        )

    # Compute probabilities in batches to control memory footprint
    probs_list = []
    for offset in range(0, candidate_df.height, batch_size):
        chunk = candidate_df.slice(offset, batch_size)
        chunk_probs = model.predict_proba(chunk)
        probs_list.append(chunk_probs)

    all_probs = np.concatenate(probs_list) if probs_list else np.array([], dtype=np.float32)

    scored_df = candidate_df.with_columns(pl.Series("match_probability", all_probs))

    # Apply decision rule: score >= threshold -> MATCH, score < threshold -> NON-MATCH
    accepted = scored_df.filter(pl.col("match_probability") >= threshold).select(
        [s1_id_col, candidate_id_col, "match_probability"]
    )
    return accepted


def format_matching_results(
    accepted_matches: pl.DataFrame,
    all_s1_ids: Sequence[str] | set[str],
    output_path: Optional[str | Path] = None,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
) -> pl.DataFrame:
    """
    Constructs the official matching_results.tsv DataFrame.

    Rules enforced:
    - Exactly 2 columns: source1_entity_id, matched_entity_ids
    - Every S1 entity in `all_s1_ids` appears exactly once.
    - matched_entity_ids is comma-separated S2/S3 IDs, or empty string for singletons (0 matches).
    - Tab-separated output format when exported to file.

    Args:
        accepted_matches: DataFrame of accepted MATCH pairs.
        all_s1_ids: Full collection of test Source1 entity IDs.
        output_path: Optional file path to write matching_results.tsv.
        s1_id_col: Name of S1 ID column.
        candidate_id_col: Name of matched candidate ID column.

    Returns:
        Polars DataFrame formatted according to competition rules.
    """
    # Group accepted MATCHes per S1 ID
    if accepted_matches.height > 0:
        grouped = (
            accepted_matches.group_by(s1_id_col, maintain_order=True)
            .agg(pl.col(candidate_id_col).unique(maintain_order=True).alias("match_list"))
            .select(
                [
                    pl.col(s1_id_col).alias("source1_entity_id"),
                    pl.col("match_list").list.join(",").alias("matched_entity_ids"),
                ]
            )
        )
    else:
        grouped = pl.DataFrame(
            {
                "source1_entity_id": pl.Series([], dtype=pl.Utf8),
                "matched_entity_ids": pl.Series([], dtype=pl.Utf8),
            }
        )

    # Join with master S1 ID list to ensure every S1 entity exists
    s1_list = sorted(list(all_s1_ids)) if all_s1_ids else []
    s1_master_df = pl.DataFrame({"source1_entity_id": pl.Series(s1_list, dtype=pl.Utf8)})
    final_df = s1_master_df.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null("")
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_csv(out_file, separator="\t")
        print(f"Matching results successfully exported to: {out_file}", flush=True)

    return final_df


def _generate_sample_inference_candidates(n_entities: int = 1000):
    """Generates candidate features for standalone script execution."""
    import duckdb
    from code.business_entity_resolution.src.preprocessing.normalize_name import normalize_business_name
    from code.business_entity_resolution.src.preprocessing.normalize_address import normalize_business_address
    from code.business_entity_resolution.src.blocking.exact_blocking import generate_exact_blocks
    from code.business_entity_resolution.src.blocking.token_blocking import generate_token_blocks
    from code.business_entity_resolution.src.features.pair_features import (
        standardize_candidate_schema,
        hydrate_candidate_pairs,
        extract_pair_features,
    )

    train_dir = PROJECT_ROOT / "student_resource" / "dataset" / "train"
    s1_path = (train_dir / "train_source1.tsv").as_posix()
    s2_path = (train_dir / "train_source2.tsv").as_posix()
    s3_path = (train_dir / "train_source3.tsv").as_posix()

    con = duckdb.connect()
    s1_df_raw = con.execute(f"SELECT * FROM read_csv_auto('{s1_path}', delim='\t') LIMIT {n_entities}").df()
    s2_df_raw = con.execute(f"SELECT * FROM read_csv_auto('{s2_path}', delim='\t') LIMIT 15000").df()
    s3_df_raw = con.execute(f"SELECT * FROM read_csv_auto('{s3_path}', delim='\t') LIMIT 15000").df()
    con.close()

    s1_df_raw["name_normalized"] = s1_df_raw["business_name"].apply(normalize_business_name)
    s1_df_raw["address_normalized"] = s1_df_raw["business_address"].apply(normalize_business_address)
    s2_df_raw["name_normalized"] = s2_df_raw["business_name"].apply(normalize_business_name)
    s2_df_raw["address_normalized"] = s2_df_raw["business_address"].apply(normalize_business_address)
    s3_df_raw["name_normalized"] = s3_df_raw["business_name"].apply(normalize_business_name)
    s3_df_raw["address_normalized"] = s3_df_raw["business_address"].apply(normalize_business_address)

    s1_prep = s1_df_raw.rename(columns={"entity_id": "s1_id"})[["s1_id", "country", "name_normalized", "address_normalized"]]
    s2_prep = s2_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]
    s3_prep = s3_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]

    exact_s2 = generate_exact_blocks(s1_prep, s2_prep, key_column="name_normalized")
    exact_s3 = generate_exact_blocks(s1_prep, s3_prep, key_column="name_normalized")
    exact_s2["source"] = "S2"
    exact_s3["source"] = "S3"
    exact_df = pd.concat([exact_s2, exact_s3], ignore_index=True).drop_duplicates()
    exact_df["block_type"] = "exact"

    token_s2 = generate_token_blocks(s1_prep, s2_prep, max_token_frequency=100, min_token_length=3)
    token_s3 = generate_token_blocks(s1_prep, s3_prep, max_token_frequency=100, min_token_length=3)
    token_s2["source"] = "S2"
    token_s3["source"] = "S3"
    token_df = pd.concat([token_s2, token_s3], ignore_index=True).drop_duplicates()
    token_df["block_type"] = "token"

    all_cands = pd.concat([exact_df, token_df], ignore_index=True)
    grouped_cands = (
        all_cands.groupby(["s1_id", "matched_id", "source"], as_index=False)
        .agg(
            block_type=("block_type", lambda x: ",".join(sorted(set(x)))),
            number_of_blocks_hit=("block_type", "nunique"),
        )
    )

    std_df = standardize_candidate_schema(grouped_cands)
    s1_polars = pl.from_pandas(s1_df_raw)
    s2_polars = pl.from_pandas(s2_df_raw)
    s3_polars = pl.from_pandas(s3_df_raw)

    hydrated_df = hydrate_candidate_pairs(std_df, s1_polars, s2_polars, s3_polars)
    features_df = extract_pair_features(hydrated_df)

    all_s1_ids = s1_df_raw["entity_id"].unique().tolist()
    return features_df, all_s1_ids


def run_prediction(
    model_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    candidate_df: Optional[pl.DataFrame] = None,
    all_s1_ids: Optional[Sequence[str] | set[str]] = None,
    output_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> pl.DataFrame:
    """
    Complete inference pipeline:
    1. Load trained model artifact and threshold configuration
    2. Score candidate pairs using score >= threshold -> MATCH, score < threshold -> NON-MATCH
    3. Format matching results and write output/matching_results.tsv

    Args:
        model_path: Path to entity_matcher_lgbm.joblib model artifact.
        config_path: Path to entity_matcher_threshold_config.json.
        candidate_df: Optional candidate pairs DataFrame with features.
        all_s1_ids: Optional list/set of all S1 IDs to include.
        output_path: Destination path for matching_results.tsv.
        threshold: Optional override for decision threshold.

    Returns:
        Formatted matching results Polars DataFrame.
    """
    repo_root = PROJECT_ROOT
    if model_path is None:
        model_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "entity_matcher_lgbm.joblib"
    if config_path is None:
        config_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "entity_matcher_threshold_config.json"
    if output_path is None:
        output_path = repo_root / "output" / "matching_results.tsv"

    model_file = Path(model_path)
    output_tsv = Path(output_path)

    # Determine optimal decision threshold
    optimal_threshold = 0.50
    if threshold is not None:
        optimal_threshold = threshold
    elif Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            optimal_threshold = float(cfg.get("best_threshold", 0.50))

    print("=" * 70, flush=True)
    print("RUNNING INFERENCE & GENERATING MATCHING PREDICTIONS", flush=True)
    print("=" * 70, flush=True)
    print(f"Loading trained model from:          {model_path}", flush=True)
    
    if not model_file.exists():
        print(f"Model artifact not found at {model_file}. Launching model training pipeline...", flush=True)
        from code.business_entity_resolution.src.models.train_model import _run_full_training_pipeline
        _run_full_training_pipeline()
        return pl.DataFrame()

    model = EntityMatcherModel.load(model_file)
    print(f"Applied Optimal Decision Threshold:  {optimal_threshold:.4f} (score >= {optimal_threshold:.4f} -> MATCH)", flush=True)

    if candidate_df is None or candidate_df.height == 0:
        print("Generating candidate pairs & 29 pairwise similarity features...", flush=True)
        candidate_df, all_s1_ids = _generate_sample_inference_candidates(n_entities=1000)

    if all_s1_ids is None:
        if candidate_df.height > 0 and "source1_entity_id" in candidate_df.columns:
            all_s1_ids = candidate_df["source1_entity_id"].unique().to_list()
        else:
            all_s1_ids = []

    # Score and filter matches
    accepted_matches = predict_matches_batch(
        model=model,
        candidate_df=candidate_df,
        threshold=optimal_threshold,
    )

    # Format matching results and export TSV
    final_results = format_matching_results(
        accepted_matches=accepted_matches,
        all_s1_ids=all_s1_ids,
        output_path=output_tsv,
    )

    singletons = final_results.filter(pl.col("matched_entity_ids") == "").height
    total_eval = final_results.height
    print(f"\nInference Summary:", flush=True)
    print(f"  Total S1 Entities Processed:        {total_eval:,}", flush=True)
    print(f"  Total Accepted Match Pairs:         {accepted_matches.height:,}", flush=True)
    print(f"  Singleton Entities (0 Matches):      {singletons:,} ({singletons / total_eval * 100:.2f}%)" if total_eval > 0 else "  Singleton Entities: 0", flush=True)
    print("=" * 70, flush=True)

    return final_results


if __name__ == "__main__":
    run_prediction()
