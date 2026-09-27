"""
Entity-level Macro evaluation, precision/recall metrics, and candidate distribution statistics.

Implements the official Amazon ML Challenge 2026 scoring rules:
Precision, Recall, F1, F0.5 (minimizing false positives), Accuracy, False Positives, and False Negatives.
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
    Computes precision, recall, F1, F_beta (F0.5), and error counts for a single Source1 entity.

    Rules:
    - If true_ids is empty and pred_ids is empty: Precision=1.0, Recall=1.0, F=1.0 (correct singleton).
    - If true_ids is empty and pred_ids is NOT empty: Precision=0.0, Recall=1.0 (or 0.0), F=0.0 (false merge).
    - If true_ids is NOT empty and pred_ids is empty: Precision=1.0 (or 0.0), Recall=0.0, F=0.0 (missed match).
    - If both non-empty:
        Precision = |True ∩ Pred| / |Pred|
        Recall    = |True ∩ Pred| / |True|
        F1        = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
        F_beta    = ((1 + beta^2) * P * R) / (beta^2 * P + R) if (beta^2 * P + R) > 0 else 0.0

    Args:
        true_ids: Ground-truth matched S2/S3 entity IDs.
        pred_ids: Predicted matched S2/S3 entity IDs.
        beta: F-score beta parameter (0.5 for precision priority / false positive minimization).

    Returns:
        Dictionary of single-entity metrics including TP, FP, FN counts.
    """
    beta_sq = beta ** 2
    is_singleton = len(true_ids) == 0

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if len(true_ids) == 0 and len(pred_ids) == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "f_score": 1.0,
            "f05_score": 1.0,
            "exact_match": 1.0,
            "is_singleton": 1.0,
            "singleton_correct": 1.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }
    elif len(true_ids) == 0 and len(pred_ids) > 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "f_score": 0.0,
            "f05_score": 0.0,
            "exact_match": 0.0,
            "is_singleton": 1.0,
            "singleton_correct": 0.0,
            "tp": 0.0,
            "fp": float(fp),
            "fn": 0.0,
        }
    elif len(true_ids) > 0 and len(pred_ids) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "f_score": 0.0,
            "f05_score": 0.0,
            "exact_match": 0.0,
            "is_singleton": 0.0,
            "singleton_correct": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(fn),
        }
    else:
        prec = float(tp / len(pred_ids))
        rec = float(tp / len(true_ids))

        f1_denom = prec + rec
        f1_val = float(2.0 * prec * rec / f1_denom) if f1_denom > 0.0 else 0.0

        f05_denom = (beta_sq * prec) + rec
        f05_val = float((1.0 + beta_sq) * prec * rec / f05_denom) if f05_denom > 0.0 else 0.0

        exact = 1.0 if true_ids == pred_ids else 0.0

        return {
            "precision": prec,
            "recall": rec,
            "f1_score": f1_val,
            "f_score": f05_val,
            "f05_score": f05_val,
            "exact_match": exact,
            "is_singleton": 0.0,
            "singleton_correct": 0.0,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }


def _parse_mapping(
    data: Union[Mapping[str, set[str]], pl.DataFrame, str, Path],
    id_col: str = "source1_entity_id",
    match_col: str = "matched_entity_ids",
) -> dict[str, set[str]]:
    """Helper to convert TSV path, DataFrame, or dict into {s1_id: set(matched_ids)}."""
    if isinstance(data, dict):
        return {k: set(v) if isinstance(v, (set, list, tuple)) else ({v} if v else set()) for k, v in data.items()}
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
        if val is None or val == "" or str(val).lower() == "null" or str(val).lower() == "nan":
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
    Computes Precision, Recall, F1, F0.5 (minimizing false positives), Accuracy,
    False Positives, and False Negatives.

    Args:
        ground_truth: Ground truth mapping or path.
        predictions: Predictions mapping or path.
        beta: F-beta parameter (default: 0.5 for false positive penalty).
        all_s1_ids: Optional explicit set of all S1 IDs to evaluate over.

    Returns:
        Dictionary containing:
        - precision / macro_precision: Precision score
        - recall / macro_recall: Recall score
        - f1 / macro_f1: F1 score
        - f05 / macro_f05: F0.5 score (primary metric)
        - accuracy / exact_set_match_rate: Accuracy (exact set match rate)
        - false_positives: Total false positive prediction count
        - false_negatives: Total false negative prediction count
        - true_positives: Total true positive prediction count
        - singleton_accuracy: Accuracy on ground truth singletons (0 matches)
        - avg_predicted_matches: Mean predicted matches per S1
        - avg_true_matches: Mean true matches per S1
        - total_entities: Number of evaluated S1 entities
        - singleton_entities: Number of ground-truth singletons
    """
    gt_map = _parse_mapping(ground_truth)
    pred_map = _parse_mapping(predictions)

    eval_s1_ids = list(all_s1_ids) if all_s1_ids is not None else list(set(gt_map.keys()) | set(pred_map.keys()))

    if not eval_s1_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f05": 0.0,
            "accuracy": 0.0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "singleton_accuracy": 0.0,
            "exact_set_match_rate": 0.0,
            "total_entities": 0,
            "singleton_entities": 0,
        }

    f05_scores = []
    f1_scores = []
    precisions = []
    recalls = []
    exact_matches = []
    singleton_flags = []
    singleton_corrects = []
    pred_counts = []
    true_counts = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s1 in eval_s1_ids:
        t_ids = gt_map.get(s1, set())
        p_ids = pred_map.get(s1, set())

        m = compute_single_entity_metrics(t_ids, p_ids, beta=beta)
        f05_scores.append(m["f05_score"])
        f1_scores.append(m["f1_score"])
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        exact_matches.append(m["exact_match"])
        singleton_flags.append(m["is_singleton"])
        singleton_corrects.append(m["singleton_correct"])
        pred_counts.append(len(p_ids))
        true_counts.append(len(t_ids))
        total_tp += int(m["tp"])
        total_fp += int(m["fp"])
        total_fn += int(m["fn"])

    n_singletons = sum(singleton_flags)
    singleton_acc = (sum(singleton_corrects) / n_singletons) if n_singletons > 0 else 1.0
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    macro_f05 = float(np.mean(f05_scores))
    exact_match_rate = float(np.mean(exact_matches))

    return {
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "f05": macro_f05,
        "accuracy": exact_match_rate,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "true_positives": total_tp,
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "singleton_accuracy": float(singleton_acc),
        "exact_set_match_rate": exact_match_rate,
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
