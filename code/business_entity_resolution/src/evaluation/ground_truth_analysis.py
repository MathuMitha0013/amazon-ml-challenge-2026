"""
Ground truth statistical profiling and match distribution analysis.

Analyzes training ground truth to profile match distributions, singleton ratios,
S2 vs S3 frequency breakdowns, and country alignment patterns.
"""

from __future__ import annotations

from typing import Any, Mapping, Union
from pathlib import Path
import numpy as np
import polars as pl

from .evaluate import _parse_mapping

    # Load Ground Truth
    gt_df = pl.read_csv(gt_path, separator="\t")

def analyze_ground_truth(
    ground_truth: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
) -> dict[str, Any]:
    """
    Computes summary distribution metrics from ground truth dataset.

    Args:
        ground_truth: Ground truth mapping, DataFrame, or TSV path.

    Returns:
        Dictionary containing:
        - total_entities: Number of master S1 entities
        - singleton_count: Number of S1 entities with 0 matches
        - singleton_ratio: Fraction of entities with 0 matches
        - total_matches: Total number of positive (S1, S2/S3) match links
        - s2_matches: Count of matched S2 entities
        - s3_matches: Count of matched S3 entities
        - match_count_mean: Mean matches per entity
        - match_count_median: Median matches per entity
        - match_count_p95: 95th percentile matches per entity
        - match_count_max: Maximum matches for any single S1 entity
    """
    gt_map = _parse_mapping(ground_truth)
    total_entities = len(gt_map)
    if total_entities == 0:
        return {"total_entities": 0, "singleton_count": 0, "singleton_ratio": 0.0}

    match_counts = []
    s2_count = 0
    s3_count = 0
    singleton_count = 0

    for s1, ids in gt_map.items():
        n = len(ids)
        match_counts.append(n)
        if n == 0:
            singleton_count += 1
        for cid in ids:
            c_upper = cid.upper()
            if c_upper.startswith("S2-"):
                s2_count += 1
            elif c_upper.startswith("S3-"):
                s3_count += 1

    counts_arr = np.array(match_counts)

    return {
        "total_entities": total_entities,
        "singleton_count": singleton_count,
        "singleton_ratio": float(singleton_count / total_entities),
        "total_matches": int(np.sum(counts_arr)),
        "s2_matches": s2_count,
        "s3_matches": s3_count,
        "match_count_mean": float(np.mean(counts_arr)),
        "match_count_median": float(np.median(counts_arr)),
        "match_count_p95": float(np.percentile(counts_arr, 95)),
        "match_count_max": int(np.max(counts_arr)),
    }
