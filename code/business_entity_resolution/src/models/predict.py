"""
Batch inference, prediction scoring, threshold gating, and submission formatting.

Generates final predictions from candidate pairs and produces format-compliant matching_results.tsv.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence
import polars as pl
import numpy as np

from .train_model import EntityMatcherModel


def predict_matches_batch(
    model: EntityMatcherModel,
    candidate_df: pl.DataFrame,
    threshold: float = 0.5,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
    batch_size: int = 200_000,
) -> pl.DataFrame:
    """
    Computes match probabilities for candidate pairs and filters pairs exceeding decision threshold.

    Args:
        model: Fitted EntityMatcherModel.
        candidate_df: Candidate pairs DataFrame with extracted features.
        threshold: Decision cutoff probability (typically tuned via validation Macro F0.5).
        s1_id_col: Source1 ID column name.
        candidate_id_col: Candidate ID column name.
        batch_size: Chunk size for scoring.

    Returns:
        Polars DataFrame of accepted matches with columns [s1_id_col, candidate_id_col, 'match_probability'].
    """
    if candidate_df.height == 0:
        return pl.DataFrame(
            {
                s1_id_col: pl.Series([], dtype=pl.Utf8),
                candidate_id_col: pl.Series([], dtype=pl.Utf8),
                "match_probability": pl.Series([], dtype=pl.Float32),
            }
        )

    # Compute probabilities in batches to control memory
    probs_list = []
    for offset in range(0, candidate_df.height, batch_size):
        chunk = candidate_df.slice(offset, batch_size)
        chunk_probs = model.predict_proba(chunk)
        probs_list.append(chunk_probs)

    all_probs = np.concatenate(probs_list) if probs_list else np.array([], dtype=np.float32)

    scored_df = candidate_df.with_columns(pl.Series("match_probability", all_probs))

    # Filter above threshold
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
    - matched_entity_ids is comma-separated S2/S3 IDs, or empty for singletons.
    - No duplicate IDs within any row list.
    - Tab-separated output when written to file.

    Args:
        accepted_matches: DataFrame of accepted match pairs.
        all_s1_ids: Full collection of test Source1 entity IDs.
        output_path: Optional file path to write matching_results.tsv.
        s1_id_col: Name of S1 ID column.
        candidate_id_col: Name of matched candidate ID column.

    Returns:
        Polars DataFrame formatted according to competition rules.
    """
    # Group accepted matches per S1 ID
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
    s1_master_df = pl.DataFrame({"source1_entity_id": sorted(list(all_s1_ids))})
    final_df = s1_master_df.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null("")
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_csv(out_file, separator="\t")

    return final_df
