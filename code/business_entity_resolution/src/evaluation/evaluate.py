"""
Entity-level Macro F0.5 evaluation and candidate distribution statistics.

Implements the official Amazon ML Challenge 2026 scoring rules:
Precision-heavy F0.5 score evaluated per Source1 entity and averaged macro-style.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union
from pathlib import Path
import numpy as np
import polars as pl


def compute_single_entity_metrics(
    true_ids: set[str],
    pred_ids: set[str],
    beta: float = 0.5,
) -> dict[str, float]:
    """
    Computes precision, recall, and F_beta for a single Source1 entity.

    Rules:
    - If true_ids is empty and pred_ids is empty: Precision=1.0, Recall=1.0, F=1.0 (correct singleton).
    - If true_ids is empty and pred_ids is NOT empty: Precision=0.0, Recall=1.0 (or 0.0), F=0.0 (false merge).
    - If true_ids is NOT empty and pred_ids is empty: Precision=1.0 (or 0.0), Recall=0.0, F=0.0 (missed match).
    - If both non-empty:
        Precision = |True ∩ Pred| / |Pred|
        Recall    = |True ∩ Pred| / |True|
        F_beta    = ((1 + beta^2) * P * R) / (beta^2 * P + R) if (beta^2 * P + R) > 0 else 0.0

    Args:
        true_ids: Ground-truth matched S2/S3 entity IDs.
        pred_ids: Predicted matched S2/S3 entity IDs.
        beta: F-score beta parameter (0.5 for precision priority).

    Returns:
        Dictionary of single-entity metrics: {'precision', 'recall', 'f_score', 'exact_match', 'is_singleton'}
    """
    beta_sq = beta ** 2
    is_singleton = len(true_ids) == 0

    if len(true_ids) == 0 and len(pred_ids) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f_score": 1.0,
            "exact_match": 1.0,
            "is_singleton": 1.0,
            "singleton_correct": 1.0,
        }
    elif len(true_ids) == 0 and len(pred_ids) > 0:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f_score": 0.0,
            "exact_match": 0.0,
            "is_singleton": 1.0,
            "singleton_correct": 0.0,
        }
    elif len(true_ids) > 0 and len(pred_ids) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0,
            "exact_match": 0.0,
            "is_singleton": 0.0,
            "singleton_correct": 0.0,
        }
    else:
        inter = len(true_ids & pred_ids)
        prec = float(inter / len(pred_ids))
        rec = float(inter / len(true_ids))
        denom = (beta_sq * prec) + rec
        f_val = float((1.0 + beta_sq) * prec * rec / denom) if denom > 0.0 else 0.0
        exact = 1.0 if true_ids == pred_ids else 0.0

        return {
            "precision": prec,
            "recall": rec,
            "f_score": f_val,
            "exact_match": exact,
            "is_singleton": 0.0,
            "singleton_correct": 0.0,
        }


def _parse_mapping(
    data: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    id_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
) -> dict[str, set[str]]:
    """Helper to convert TSV path, DataFrame, or dict into {s1_id: set(matched_ids)}."""
    if isinstance(data, dict):
        return {k: set(v) if isinstance(v, (set, list, tuple)) else {v} for k, v in data.items()}
    elif isinstance(data, (str, Path)):
        df = pl.read_csv(data, separator="\t")
    elif isinstance(data, pl.DataFrame):
        df = data
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    mapping = {}
    for row in df.iter_rows(named=True):
        s1 = str(row[id_col]).strip()
        val = row.get(match_col)
        if val is None or val == "" or str(val).lower() == "null":
            mapping[s1] = set()
        else:
            ids = {x.strip() for x in str(val).split(",") if x.strip()}
            mapping[s1] = ids
    return mapping


def evaluate_predictions(
    ground_truth: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    predictions: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    beta: float = 0.5,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """
    Computes comprehensive entity-level evaluation metrics.

    Args:
        ground_truth: Ground truth mapping or path.
        predictions: Predictions mapping or path.
        beta: F-beta parameter (default: 0.5).
        all_s1_ids: Optional explicit set of all S1 IDs to evaluate over.

    Returns:
        Dictionary containing:
        - macro_f05: Overall Macro F0.5 score
        - macro_precision: Macro precision
        - macro_recall: Macro recall
        - singleton_accuracy: Accuracy on ground truth singletons (0 matches)
        - exact_set_match_rate: Fraction of S1 entities with 100% exact set match
        - avg_predicted_matches: Mean predicted matches per S1
        - avg_true_matches: Mean true matches per S1
        - total_entities: Number of evaluated S1 entities
        - singleton_entities: Number of ground-truth singletons
    """
    gt_map = _parse_mapping(ground_truth)
    pred_map = _parse_mapping(predictions)

    eval_s1_ids = list(all_s1_ids) if all_s1_ids is not None else list(set(gt_map.keys()) | set(pred_map.keys()))

    if not eval_s1_ids:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "total_entities": 0}

    f_scores = []
    precisions = []
    recalls = []
    exact_matches = []
    singleton_flags = []
    singleton_corrects = []
    pred_counts = []
    true_counts = []

    for s1 in eval_s1_ids:
        t_ids = gt_map.get(s1, set())
        p_ids = pred_map.get(s1, set())

        m = compute_single_entity_metrics(t_ids, p_ids, beta=beta)
        f_scores.append(m["f_score"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        exact_matches.append(m["exact_match"])
        singleton_flags.append(m["is_singleton"])
        singleton_corrects.append(m["singleton_correct"])
        pred_counts.append(len(p_ids))
        true_counts.append(len(t_ids))

    n_singletons = sum(singleton_flags)
    singleton_acc = (sum(singleton_corrects) / n_singletons) if n_singletons > 0 else 1.0

    return {
        "macro_f05": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(singleton_acc),
        "exact_set_match_rate": float(np.mean(exact_matches)),
        "avg_predicted_matches": float(np.mean(pred_counts)),
        "avg_true_matches": float(np.mean(true_counts)),
        "total_entities": len(eval_s1_ids),
        "singleton_entities": int(n_singletons),
    }


def compute_candidate_statistics(
    candidates_df: pl.DataFrame,
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
) -> dict[str, float]:
    """
    Calculates candidate volume distribution statistics per Source1 entity.

    Args:
        candidates_df: DataFrame of candidate pairs.
        s1_id_col: Source1 ID column name.
        candidate_id_col: Candidate ID column name.

    Returns:
        Dictionary containing mean, median, P95, P99, min, max candidates per S1.
    """
    if candidates_df.height == 0:
        return {
            "avg_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p95_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "max_candidates_per_s1": 0.0,
            "total_pairs": 0,
        }

    counts = (
        candidates_df.group_by(s1_id_col)
        .agg(pl.count(candidate_id_col).alias("num_cands"))["num_cands"]
        .to_numpy()
    )

    return {
        "avg_candidates_per_s1": float(np.mean(counts)),
        "median_candidates_per_s1": float(np.median(counts)),
        "p95_candidates_per_s1": float(np.percentile(counts, 95)),
        "p99_candidates_per_s1": float(np.percentile(counts, 99)),
        "max_candidates_per_s1": float(np.max(counts)),
        "total_pairs": candidates_df.height,
    }


def compute_macro_f05(
    ground_truth: Any,
    predictions: Any,
    **kwargs: Any,
) -> dict[str, float]:
    """Convenience alias for evaluate_predictions."""
    return evaluate_predictions(ground_truth, predictions, **kwargs)
