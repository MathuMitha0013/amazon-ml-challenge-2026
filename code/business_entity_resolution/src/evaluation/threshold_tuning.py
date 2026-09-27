"""
Decision threshold optimization targeting validation Macro F0.5 score.

Iterates through probability thresholds from 0.30 to 0.95 in steps of 0.01
to identify the exact cutoff that maximizes the F0.5 score.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import numpy as np
import polars as pl

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.evaluation.evaluate import evaluate_predictions


def tune_f05_threshold(
    val_scored_candidates: pl.DataFrame,
    val_ground_truth: Union[Mapping[str, set[str]], pl.DataFrame, str],
    prob_col: str = "match_probability",
    s1_id_col: str = "source1_entity_id",
    candidate_id_col: str = "candidate_entity_id",
    threshold_min: float = 0.30,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """
    Performs grid search over candidate match probabilities (from threshold_min to threshold_max
    in steps of threshold_step) to find the exact cutoff that maximizes the Macro F0.5 score.

    Default search range: 0.30 to 0.95 in steps of 0.01.

    Args:
        val_scored_candidates: Validation candidate pairs with predicted match probabilities.
        val_ground_truth: Ground truth mapping or DataFrame for validation S1 entities.
        prob_col: Name of column containing predicted match probability.
        s1_id_col: Source1 ID column name.
        candidate_id_col: Candidate ID column name.
        threshold_min: Lower bound for threshold search (default: 0.30).
        threshold_max: Upper bound for threshold search (default: 0.95).
        threshold_step: Step size between search thresholds (default: 0.01).
        all_s1_ids: Complete set of validation Source1 entity IDs.

    Returns:
        Dictionary containing:
        - best_threshold: Exact probability cutoff that maximizes F0.5 score
        - best_f05: Highest achieved validation F0.5 score
        - best_macro_f05: Highest achieved validation Macro F0.5 score
        - best_metrics: Full evaluation metrics dictionary at best threshold
        - tuning_history: List of evaluation metrics across all tested thresholds
    """
    thresholds = np.arange(threshold_min, threshold_max + 1e-5, threshold_step)
    best_threshold = 0.50
    best_f05 = -1.0
    best_metrics: dict[str, Any] = {}
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

        # Filter accepted candidate pairs: score >= threshold -> MATCH
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

        current_f05 = eval_res.get("f05", eval_res.get("macro_f05", 0.0))

        record = {
            "threshold": t_val,
            "f05": current_f05,
            "macro_f05": eval_res.get("macro_f05", current_f05),
            "precision": eval_res.get("precision", 0.0),
            "recall": eval_res.get("recall", 0.0),
            "f1": eval_res.get("f1", 0.0),
            "accuracy": eval_res.get("accuracy", 0.0),
            "false_positives": eval_res.get("false_positives", 0),
            "false_negatives": eval_res.get("false_negatives", 0),
            "true_positives": eval_res.get("true_positives", 0),
            "singleton_accuracy": eval_res.get("singleton_accuracy", 0.0),
            "exact_set_match_rate": eval_res.get("exact_set_match_rate", 0.0),
            "avg_predicted_matches": eval_res.get("avg_predicted_matches", 0.0),
        }
        history.append(record)

        if current_f05 > best_f05:
            best_f05 = current_f05
            best_threshold = t_val
            best_metrics = eval_res

    return {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "best_macro_f05": best_f05,
        "best_metrics": best_metrics,
        "tuning_history": history,
    }


def main():
    """Main script entry point for running threshold tuning and saving optimal threshold configuration."""
    print("=" * 105, flush=True)
    print("THRESHOLD TUNING & PROBABILITY CUTOFF OPTIMIZATION (0.30 to 0.95, step=0.01)", flush=True)
    print("=" * 105, flush=True)

    models_dir = PROJECT_ROOT / "code" / "business_entity_resolution" / "experiments" / "models"
    model_path = models_dir / "entity_matcher_lgbm.joblib"
    config_path = models_dir / "entity_matcher_threshold_config.json"

    # If model is not trained yet, run full training pipeline to generate model and validation candidates
    if not model_path.exists():
        print(f"Model artifact not found at {model_path}. Launching model training pipeline...", flush=True)
        from code.business_entity_resolution.src.models.train_model import _run_full_training_pipeline
        _run_full_training_pipeline()
        return

    # If model exists, load threshold config and report metrics
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            best_threshold = cfg.get("best_threshold", 0.50)
            metrics = cfg.get("validation_metrics", {})
    else:
        best_threshold = 0.50
        metrics = {}

    print(f"\nModel Artifact: {model_path}", flush=True)
    print(f"Config File:    {config_path}", flush=True)
    print("\n" + "=" * 105, flush=True)
    print("OPTIMAL THRESHOLD GRID SEARCH SUMMARY")
    print("=" * 105, flush=True)
    print(f"Optimal Decision Threshold Cutoff: {best_threshold:.4f}", flush=True)
    print(f"Maximized Validation Macro F0.5:   {metrics.get('macro_f05', 0.0):.4f}", flush=True)
    print(f"Macro Precision:                   {metrics.get('macro_precision', 0.0):.4f}", flush=True)
    print(f"Macro Recall:                      {metrics.get('macro_recall', 0.0):.4f}", flush=True)
    print(f"Singleton / No-Match Accuracy:     {metrics.get('singleton_accuracy', 0.0):.2%}", flush=True)
    print(f"Exact Set Match Rate:              {metrics.get('exact_set_match_rate', 0.0):.2%}", flush=True)
    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
