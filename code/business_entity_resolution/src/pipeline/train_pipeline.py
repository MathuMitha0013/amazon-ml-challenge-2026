"""
End-to-end training and threshold calibration pipeline orchestrator.

Orchestrates candidate scoring, pairwise feature extraction, LightGBM model training,
entity-disjoint validation, and Macro F0.5 decision threshold calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional
import polars as pl
import joblib

from ..features.pair_features import extract_batch_features
from ..models.train_model import EntityMatcherModel, split_entity_disjoint
from ..evaluation.threshold_tuning import tune_f05_threshold
from ..evaluation.evaluate import evaluate_predictions, compute_candidate_statistics


def train_and_calibrate_pipeline(
    candidate_pairs_df: pl.DataFrame,
    ground_truth: Any,
    output_model_dir: str | Path = "models_saved",
    top_k_candidates: int = 30,
    model_params: Optional[dict[str, Any]] = None,
    val_split_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Executes training on candidate pairs with pairwise feature extraction and validation threshold tuning.

    Args:
        candidate_pairs_df: DataFrame of candidate pairs with text attributes.
        ground_truth: Ground truth mapping or path.
        output_model_dir: Directory to save trained model artifact and optimal threshold config.
        top_k_candidates: Maximum candidates per S1 entity.
        model_params: Hyperparameters for LightGBM.
        val_split_size: Fraction of S1 entities held out for validation.
        random_seed: Random seed.

    Returns:
        Dictionary of training and evaluation results.
    """
    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Split candidate pairs into train and validation grouped by Source1 entity
    train_cands, val_cands = split_entity_disjoint(
        candidate_pairs_df,
        s1_id_col="source1_entity_id",
        test_size=val_split_size,
        random_state=random_seed,
    )

    # 2. Extract pairwise features
    train_feats = extract_batch_features(train_cands)
    val_feats = extract_batch_features(val_cands)

    # 3. Train LightGBM EntityMatcherModel
    y_train = train_feats["label"].to_numpy()
    y_val = val_feats["label"].to_numpy() if "label" in val_feats.columns else None

    params = model_params or {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_seed,
    }

    model = EntityMatcherModel(**params)
    model.fit(train_feats, y_train, X_val=val_feats, y_val=y_val)

    # 4. Save trained model
    model_file = out_dir / "lgbm_matcher.joblib"
    model.save(model_file)

    # 5. Tune threshold on validation set
    val_probs = model.predict_proba(val_feats)
    val_scored = val_feats.with_columns(pl.Series("match_probability", val_probs))

    tuning_res = tune_f05_threshold(
        val_scored_candidates=val_scored,
        val_ground_truth=ground_truth,
        threshold_min=0.10,
        threshold_max=0.90,
        threshold_step=0.02,
    )

    best_threshold = tuning_res["best_threshold"]

    # Save threshold metadata
    config_file = out_dir / "threshold_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_threshold": best_threshold,
                "best_macro_f05": tuning_res["best_macro_f05"],
                "best_metrics": tuning_res["best_metrics"],
                "top_k_candidates": top_k_candidates,
            },
            f,
            indent=2,
        )

    return {
        "model_path": str(model_file),
        "config_path": str(config_file),
        "best_threshold": best_threshold,
        "validation_metrics": tuning_res["best_metrics"],
    }


def run_training_pipeline(
    train_dir: str | Path = "dataset/train",
    output_model_dir: str | Path = "models_saved",
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    CLI wrapper for end-to-end training pipeline.
    """
    from code.business_entity_resolution.experiments.train_4route_model import run_4route_training
    print(f"Executing training pipeline on {train_dir}...", flush=True)
    run_4route_training()
    return {
        "status": "success",
        "output_model_dir": str(output_model_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Training Pipeline.")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training dataset folder.")
    parser.add_argument("--output-model-dir", default="models_saved", help="Path to save trained models.")
    args = parser.parse_args()

    print("Running training pipeline...")
    run_training_pipeline(args.train_dir, args.output_model_dir)


if __name__ == "__main__":
    main()
