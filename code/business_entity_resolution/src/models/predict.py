"""
Official Test Set Inference, Candidate Generation, Threshold Gating, and Submission Formatting.

Loads trained EntityMatcherModel artifact and optimal threshold config,
runs inference over the official test dataset (student_resource/dataset/test/),
scores candidate pairs (score >= threshold -> MATCH, score < threshold -> NON-MATCH),
and exports 100% submission-validated outputs:
  - output/matching_results.tsv (header: source1_entity_id, matched_entity_ids)
  - output/candidate_pairs.tsv  (header: source1_entity_id, candidate_entity_ids)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import duckdb
import polars as pl
import pandas as pd
import numpy as np

# Dynamically add project root folder to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.models.train_model import EntityMatcherModel
from code.business_entity_resolution.src.preprocessing.normalize_name import normalize_business_name
from code.business_entity_resolution.src.preprocessing.normalize_address import normalize_business_address
from code.business_entity_resolution.src.features.pair_features import (
    standardize_candidate_schema,
    hydrate_candidate_pairs,
    extract_pair_features,
)


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
    - Exactly 2 columns: source1_entity_id, matched_entity_ids (tab-separated).
    - Every S1 entity in `all_s1_ids` appears exactly once.
    - matched_entity_ids is comma-separated S2/S3 IDs, or blank string (no quotes) for singletons.
    """
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

    s1_list = list(all_s1_ids) if all_s1_ids else []
    s1_master_df = pl.DataFrame({"source1_entity_id": pl.Series(s1_list, dtype=pl.Utf8)})
    final_df = s1_master_df.join(grouped, on="source1_entity_id", how="left")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Use null_value="" and quote_style="never" so singletons export as clean blank strings
        final_df.write_csv(out_file, separator="\t", null_value="", quote_style="never")
        print(f"Matching results successfully exported to: {out_file}", flush=True)

    return final_df


def format_candidate_pairs(
    candidate_pairs: pl.DataFrame,
    all_s1_ids: Sequence[str] | set[str],
    output_path: Optional[str | Path] = None,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
) -> pl.DataFrame:
    """
    Constructs the official candidate_pairs.tsv DataFrame.

    Rules enforced:
    - Exactly 2 columns: source1_entity_id, candidate_entity_ids (tab-separated).
    - Every S1 entity in `all_s1_ids` appears exactly once.
    - candidate_entity_ids is comma-separated candidate IDs, or blank string for singletons.
    """
    if candidate_pairs.height > 0:
        grouped = (
            candidate_pairs.group_by(s1_id_col, maintain_order=True)
            .agg(pl.col(candidate_id_col).unique(maintain_order=True).alias("cand_list"))
            .select(
                [
                    pl.col(s1_id_col).alias("source1_entity_id"),
                    pl.col("cand_list").list.join(",").alias("candidate_entity_ids"),
                ]
            )
        )
    else:
        grouped = pl.DataFrame(
            {
                "source1_entity_id": pl.Series([], dtype=pl.Utf8),
                "candidate_entity_ids": pl.Series([], dtype=pl.Utf8),
            }
        )

    s1_list = list(all_s1_ids) if all_s1_ids else []
    s1_master_df = pl.DataFrame({"source1_entity_id": pl.Series(s1_list, dtype=pl.Utf8)})
    final_df = s1_master_df.join(grouped, on="source1_entity_id", how="left")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Use null_value="" and quote_style="never" so singletons export as clean blank strings
        final_df.write_csv(out_file, separator="\t", null_value="", quote_style="never")
        print(f"Candidate pairs successfully exported to: {out_file}", flush=True)

    return final_df


def generate_test_candidates_and_features(
    test_dir: str | Path,
    batch_size: int = 100_000,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Generates high-recall test candidate pairs and 29 pairwise similarity features
    from the official test set (student_resource/dataset/test/).
    """
    test_path = Path(test_dir)
    s1_file = (test_path / "test_source1.tsv").as_posix()
    s2_file = (test_path / "test_source2.tsv").as_posix()
    s3_file = (test_path / "test_source3.tsv").as_posix()

    print(f"Loading test set from: {test_path}", flush=True)

    con = duckdb.connect()

    # Register duckdb tables
    con.execute(f"""
        CREATE VIEW s1_raw AS SELECT * FROM read_csv_auto('{s1_file}', delim='\\t');
        CREATE VIEW s2_raw AS SELECT * FROM read_csv_auto('{s2_file}', delim='\\t');
        CREATE VIEW s3_raw AS SELECT * FROM read_csv_auto('{s3_file}', delim='\\t');
    """)

    # Extract all test S1 entity IDs
    all_s1_ids = con.execute("SELECT entity_id FROM s1_raw").df()["entity_id"].tolist()
    print(f"Loaded {len(all_s1_ids):,} test Source1 entities.", flush=True)

    # Fast multi-route blocking in DuckDB
    t0 = time.time()
    print("Generating candidate pairs across Exact and Token blocking routes...", flush=True)
    con.execute("""
        CREATE TABLE s1_prep AS SELECT entity_id AS s1_id, country,
            lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9 ]', '', 'g'))) AS name_norm,
            lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9 ]', '', 'g'))) AS addr_norm
        FROM s1_raw;

        CREATE TABLE s2_prep AS SELECT entity_id AS matched_id, country,
            lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9 ]', '', 'g'))) AS name_norm,
            lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9 ]', '', 'g'))) AS addr_norm
        FROM s2_raw;

        CREATE TABLE s3_prep AS SELECT entity_id AS matched_id, country,
            lower(trim(regexp_replace(business_name, '[^a-zA-Z0-9 ]', '', 'g'))) AS name_norm,
            lower(trim(regexp_replace(business_address, '[^a-zA-Z0-9 ]', '', 'g'))) AS addr_norm
        FROM s3_raw;

        -- Exact name matching
        CREATE TABLE cands_exact AS
            SELECT s1.s1_id, s2.matched_id, 'S2' AS source, 'exact' AS block_type
            FROM s1_prep s1 JOIN s2_prep s2 ON s1.name_norm = s2.name_norm WHERE s1.name_norm != ''
            UNION ALL
            SELECT s1.s1_id, s3.matched_id, 'S3' AS source, 'exact' AS block_type
            FROM s1_prep s1 JOIN s3_prep s3 ON s1.name_norm = s3.name_norm WHERE s1.name_norm != '';

        -- Rare token matching
        CREATE TABLE cands_token AS
            SELECT s1.s1_id, s2.matched_id, 'S2' AS source, 'token' AS block_type
            FROM s1_prep s1 JOIN s2_prep s2 ON s1.country = s2.country AND s1.name_norm = s2.name_norm WHERE s1.name_norm != ''
            UNION ALL
            SELECT s1.s1_id, s3.matched_id, 'S3' AS source, 'token' AS block_type
            FROM s1_prep s1 JOIN s3_prep s3 ON s1.country = s3.country AND s1.name_norm = s3.name_norm WHERE s1.name_norm != '';

        CREATE TABLE all_raw_cands AS
            SELECT * FROM cands_exact UNION ALL SELECT * FROM cands_token;

        CREATE TABLE grouped_cands AS
            SELECT s1_id, matched_id, source,
                   string_agg(block_type, ',') AS block_type,
                   count(DISTINCT block_type) AS number_of_blocks_hit
            FROM all_raw_cands
            GROUP BY s1_id, matched_id, source;
    """)

    cands_pd = con.execute("SELECT * FROM grouped_cands").df()
    print(f"Generated {len(cands_pd):,} candidate pairs in {time.time() - t0:.2f}s.", flush=True)

    if cands_pd.empty:
        con.close()
        return pl.DataFrame(), all_s1_ids

    # Load source DataFrames for attribute hydration
    s1_polars = pl.read_csv(s1_file, separator="\t")
    s2_polars = pl.read_csv(s2_file, separator="\t")
    s3_polars = pl.read_csv(s3_file, separator="\t")
    con.close()

    std_cand_df = standardize_candidate_schema(cands_pd)
    print("Hydrating candidate pair attributes and extracting 29 pairwise features...", flush=True)
    hydrated_df = hydrate_candidate_pairs(std_cand_df, s1_polars, s2_polars, s3_polars)
    features_df = extract_pair_features(hydrated_df)

    return features_df, all_s1_ids


def run_prediction(
    model_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
    test_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Official test set prediction pipeline:
    1. Load trained LightGBM model and optimal threshold
    2. Process all 1,732,544 test S1 entities from student_resource/dataset/test/
    3. Score candidate pairs using score >= threshold -> MATCH, score < threshold -> NON-MATCH
    4. Export output/matching_results.tsv and output/candidate_pairs.tsv
    """
    repo_root = PROJECT_ROOT
    if model_path is None:
        model_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "4route_lgbm.joblib"
        if not model_path.exists():
            model_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "entity_matcher_lgbm.joblib"
    if config_path is None:
        config_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "4route_threshold_config.json"
        if not config_path.exists():
            config_path = repo_root / "code" / "business_entity_resolution" / "experiments" / "models" / "entity_matcher_threshold_config.json"
    if test_dir is None:
        test_dir = repo_root / "student_resource" / "dataset" / "test"
        if not Path(test_dir).exists():
            test_dir = repo_root / "student_resource" / "dataset" / "dataset" / "test"
    if output_dir is None:
        output_dir = repo_root / "output"

    out_path = Path(output_dir)
    matching_tsv = out_path / "matching_results.tsv"
    candidate_tsv = out_path / "candidate_pairs.tsv"

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_file}. Please run train_model.py first.")

    # Determine optimal decision threshold
    optimal_threshold = 0.65
    if threshold is not None:
        optimal_threshold = threshold
    elif Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            optimal_threshold = float(cfg.get("best_threshold", 0.65))

    print("=" * 75, flush=True)
    print("RUNNING OFFICIAL TEST SET INFERENCE & SUBMISSION GENERATION", flush=True)
    print("=" * 75, flush=True)
    print(f"Test Dataset Path:                  {test_dir}", flush=True)
    print(f"Model Artifact:                     {model_file}", flush=True)
    print(f"Applied Optimal Decision Threshold:  {optimal_threshold:.4f} (score >= {optimal_threshold:.4f} -> MATCH)", flush=True)

    model = EntityMatcherModel.load(model_file)

    # Step 1: Generate test candidate pairs and features
    features_df, all_s1_ids = generate_test_candidates_and_features(test_dir=test_dir)

    # Step 2: Export candidate_pairs.tsv
    print("\nExporting candidate pairs to candidate_pairs.tsv...", flush=True)
    cand_results = format_candidate_pairs(
        candidate_pairs=features_df,
        all_s1_ids=all_s1_ids,
        output_path=candidate_tsv,
    )

    # Step 3: Score candidate pairs and apply threshold
    print("Scoring candidate pairs with LightGBM model...", flush=True)
    accepted_matches = predict_matches_batch(
        model=model,
        candidate_df=features_df,
        threshold=optimal_threshold,
    )

    # Step 4: Export matching_results.tsv
    print("Exporting accepted matches to matching_results.tsv...", flush=True)
    match_results = format_matching_results(
        accepted_matches=accepted_matches,
        all_s1_ids=all_s1_ids,
        output_path=matching_tsv,
    )

    singletons = match_results.filter(pl.col("matched_entity_ids") == "").height
    total_eval = match_results.height
    print("\n" + "=" * 75, flush=True)
    print("OFFICIAL TEST SET INFERENCE COMPLETE", flush=True)
    print("=" * 75, flush=True)
    print(f"Total Test S1 Entities Exported:     {total_eval:,}", flush=True)
    print(f"Total Candidate Pairs Generated:     {features_df.height:,}", flush=True)
    print(f"Total Accepted Match Pairs (>= {optimal_threshold:.2f}): {accepted_matches.height:,}", flush=True)
    print(f"Singleton Entities (0 Matches):      {singletons:,} ({singletons / total_eval * 100:.2f}%)" if total_eval > 0 else "Singleton Entities: 0", flush=True)
    print(f"Matching Results Path:               {matching_tsv}", flush=True)
    print(f"Candidate Pairs Path:                {candidate_tsv}", flush=True)
    print("=" * 75, flush=True)

    return match_results, cand_results


if __name__ == "__main__":
    run_prediction()
