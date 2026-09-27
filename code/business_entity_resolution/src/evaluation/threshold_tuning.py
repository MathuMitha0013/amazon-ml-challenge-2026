"""
Decision threshold optimization targeting validation Macro F0.5.

Calibrates confidence cutoffs across validation folds using fine-grained grid search
to maximize the competition metric while penalizing false merges on singletons.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import polars as pl

from .evaluate import evaluate_predictions


def tune_f05_threshold(
    val_scored_candidates: pl.DataFrame,
    val_ground_truth: Union[Mapping[str, set[str]], pl.DataFrame, str],
    prob_col: str = "match_probability",
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.02,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """
    Performs grid search over candidate match probabilities to find the optimal
    decision threshold that maximizes entity-level Macro F0.5.

    Args:
        val_scored_candidates: Validation pairs with predicted match probabilities.
        val_ground_truth: Ground truth mapping or DataFrame for validation S1 entities.
        prob_col: Name of column containing predicted match probability.
        s1_id_col: Source1 ID column.
        candidate_id_col: Candidate ID column.
        threshold_min: Lower bound for threshold search.
        threshold_max: Upper bound for threshold search.
        threshold_step: Step size between search thresholds.
        all_s1_ids: Complete set of validation Source1 entity IDs.

    Returns:
        Dictionary containing:
        - best_threshold: Optimal probability cutoff
        - best_macro_f05: Highest achieved validation Macro F0.5
        - best_metrics: Full evaluation metrics dictionary at best threshold
        - tuning_history: List of evaluation metrics across all tested thresholds
    """
    thresholds = np.arange(threshold_min, threshold_max + 1e-5, threshold_step)
    best_threshold = 0.5
    best_f05 = -1.0
    best_metrics: dict[str, float] = {}
    history = []

    # Pre-extract numpy arrays for fast threshold filtering
    if val_scored_candidates.height > 0:
        s1_arr = val_scored_candidates[s1_id_col].to_numpy()
        c_arr = val_scored_candidates[candidate_id_col].to_numpy()
        p_arr = val_scored_candidates[prob_col].to_numpy()
    else:
        s1_arr = np.array([])
        c_arr = np.array([])
        p_arr = np.array([])

    for t in thresholds:
        t_val = round(float(t), 4)

        # Filter accepted candidate pairs
        mask = p_arr >= t_val
        accepted_s1 = s1_arr[mask]
        accepted_c = c_arr[mask]

        # Build prediction mapping {s1_id: set(cand_ids)}
        pred_map: dict[str, set[str]] = {}
        for s1, c in zip(accepted_s1, accepted_c):
            if s1 not in pred_map:
                pred_map[s1] = set()
            pred_map[s1].add(c)

        # Evaluate
        eval_res = evaluate_predictions(
            ground_truth=val_ground_truth,
            predictions=pred_map,
            beta=0.5,
            all_s1_ids=all_s1_ids,
        )

        record = {
            "threshold": t_val,
            "macro_f05": eval_res["macro_f05"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "singleton_accuracy": eval_res["singleton_accuracy"],
            "exact_set_match_rate": eval_res["exact_set_match_rate"],
            "avg_predicted_matches": eval_res["avg_predicted_matches"],
        }
        history.append(record)

        if eval_res["macro_f05"] > best_f05:
            best_f05 = eval_res["macro_f05"]
            best_threshold = t_val
            best_metrics = eval_res

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_f05,
        "best_metrics": best_metrics,
        "tuning_history": history,
    }
