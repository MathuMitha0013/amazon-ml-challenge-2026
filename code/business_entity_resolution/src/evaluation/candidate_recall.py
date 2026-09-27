"""
Candidate recall and reduction ratio evaluation.

Evaluates how effectively candidate blocking captures true ground-truth matches
and quantifies the search-space reduction achieved over the full Cartesian product.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union
from pathlib import Path
import numpy as np
import polars as pl

from .evaluate import _parse_mapping, compute_candidate_statistics


def evaluate_candidate_recall(
    ground_truth: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    candidate_pairs: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    total_s1_count: Optional[int] = None,
    total_target_count: Optional[int] = None,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
) -> dict[str, float]:
    """
    Computes ground truth match recall in the candidate pool and candidate volume metrics.

    Args:
        ground_truth: Ground truth mapping or path.
        candidate_pairs: Candidate pairs mapping, DataFrame, or path.
        total_s1_count: Optional total count of S1 records for reduction ratio.
        total_target_count: Optional total count of S2+S3 records for reduction ratio.
        s1_id_col: Source1 ID column name.
        candidate_id_col: Candidate ID column name.

    Returns:
        Dictionary containing:
        - candidate_recall: Fraction of true match pairs captured in candidate pool
        - true_matches_captured: Total count of captured positive pairs
        - true_matches_total: Total positive pairs in ground truth
        - avg_candidates_per_s1: Average number of candidate pairs per S1
        - median_candidates_per_s1: Median candidates per S1
        - p95_candidates_per_s1: 95th percentile candidates per S1
        - p99_candidates_per_s1: 99th percentile candidates per S1
        - reduction_ratio: Percentage reduction over Cartesian product (if totals provided)
    """
    gt_map = _parse_mapping(ground_truth)

    # Parse candidate pairs
    if isinstance(candidate_pairs, (dict, Mapping)):
        cand_map = {k: set(v) for k, v in candidate_pairs.items()}
        cand_df = None
    elif isinstance(candidate_pairs, pl.DataFrame):
        cand_df = candidate_pairs
        cand_map = {}
        for row in candidate_pairs.iter_rows(named=True):
            s1 = str(row[s1_id_col])
            c = str(row[candidate_id_col])
            if s1 not in cand_map:
                cand_map[s1] = set()
            cand_map[s1].add(c)
    else:
        cand_df = pl.read_csv(candidate_pairs, separator="\t")
        cand_map = _parse_mapping(cand_df, id_col=s1_id_col, match_col=candidate_id_col)

    total_true_pairs = sum(len(ids) for ids in gt_map.values())
    captured_pairs = 0

    for s1, true_ids in gt_map.items():
        if not true_ids:
            continue
        cands = cand_map.get(s1, set())
        captured_pairs += len(true_ids & cands)

    recall = float(captured_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0

    stats: dict[str, float] = {
        "candidate_recall": recall,
        "true_matches_captured": float(captured_pairs),
        "true_matches_total": float(total_true_pairs),
    }

    # Volume stats
    cand_counts = [len(cands) for cands in cand_map.values()] if cand_map else [0]
    stats["avg_candidates_per_s1"] = float(np.mean(cand_counts))
    stats["median_candidates_per_s1"] = float(np.median(cand_counts))
    stats["p95_candidates_per_s1"] = float(np.percentile(cand_counts, 95))
    stats["p99_candidates_per_s1"] = float(np.percentile(cand_counts, 99))
    stats["total_candidate_pairs"] = float(sum(cand_counts))

    if total_s1_count and total_target_count:
        full_cartesian = float(total_s1_count * total_target_count)
        cand_volume = float(sum(cand_counts))
        reduction = (1.0 - (cand_volume / full_cartesian)) * 100.0 if full_cartesian > 0 else 0.0
        stats["reduction_ratio_pct"] = reduction

    return stats
