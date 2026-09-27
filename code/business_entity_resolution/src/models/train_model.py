"""
Pairwise LightGBM matching model training, validation, and serialization.

Provides the EntityMatcherModel class with full support for entity-disjoint validation splits,
configurable hyperparameters, hard negative weighting, and early stopping.

When executed as a script (`python -m src.models.train_model` or `python src/models/train_model.py`),
runs the complete end-to-end training pipeline:
    1. Load training sources and ground truth via DuckDB
    2. Normalize text fields and generate 4-route blocking candidates
    3. Standardize, hydrate, and label candidate pairs
    4. Extract 29 pairwise similarity features
    5. Entity-disjoint train/val split (80/20)
    6. Train LightGBM binary classifier with early stopping
    7. Tune decision threshold on validation Macro F0.5
    8. Save model artifact and threshold config to experiments/models/
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

# Dynamically add project root folder to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from code.business_entity_resolution.src.features.pair_features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# EntityMatcherModel
# ---------------------------------------------------------------------------

class EntityMatcherModel:
    """
    LightGBM pairwise binary classification model for Business Entity Matching.

    Predicts whether a candidate pair (S1, S2/S3) is a true match ($y=1$) or non-match ($y=0$).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 30,
        feature_columns: Optional[Sequence[str]] = None,
        extra_params: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the EntityMatcherModel.

        Args:
            n_estimators: Maximum number of boosting trees.
            learning_rate: Boosting learning rate.
            num_leaves: Max tree leaves for base learners.
            max_depth: Maximum tree depth.
            subsample: Subsample ratio of the training instances.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            scale_pos_weight: Balancing of positive and negative weights.
            random_state: Random seed for reproducibility.
            early_stopping_rounds: Early stopping patience on validation set.
            feature_columns: Specific feature column names to train on.
            extra_params: Additional LightGBM parameters.
        """
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.early_stopping_rounds = early_stopping_rounds

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        if extra_params:
            params.update(extra_params)

        self.params = params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_iteration_: Optional[int] = None
        self.feature_importances_: Optional[dict[str, float]] = None

    def _prepare_matrix(self, data: Any) -> np.ndarray:
        """Converts Polars DataFrame, Pandas DataFrame, or NumPy array to 2D float32 matrix."""
        if isinstance(data, pl.DataFrame):
            # Select feature columns present in data
            avail_cols = [c for c in self.feature_columns if c in data.columns]
            return data.select(avail_cols).to_numpy().astype(np.float32)
        elif hasattr(data, "values"):  # pandas DataFrame
            avail_cols = [c for c in self.feature_columns if c in data.columns]
            return data[avail_cols].values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        else:
            return np.array(data, dtype=np.float32)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
    ) -> EntityMatcherModel:
        """
        Fits the LightGBM classifier.

        Args:
            X: Training features (DataFrame or matrix).
            y: Training binary labels (1 = match, 0 = non-match).
            X_val: Optional validation features.
            y_val: Optional validation binary labels.
            sample_weight: Optional sample weights (e.g. for hard negatives).

        Returns:
            self
        """
        X_mat = self._prepare_matrix(X)
        y_arr = np.asarray(y, dtype=np.int32)

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_mat = self._prepare_matrix(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.int32)
            eval_set = [(X_val_mat, y_val_arr)]
            if self.early_stopping_rounds and self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_mat,
            y_arr,
            sample_weight=sample_weight,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

        if hasattr(self.model, "best_iteration_"):
            self.best_iteration_ = self.model.best_iteration_

        # Record feature importances
        if hasattr(self.model, "feature_importances_"):
            raw_imps = self.model.feature_importances_
            self.feature_importances_ = {
                col: float(imp) for col, imp in zip(self.feature_columns, raw_imps)
            }

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predicts match probability for each candidate pair.

        Args:
            X: Feature matrix or DataFrame.

        Returns:
            1D NumPy array of probabilities for class 1 (match).
        """
        if self.model is None:
            raise RuntimeError("EntityMatcherModel must be fitted before predict_proba.")

        X_mat = self._prepare_matrix(X)
        if len(X_mat) == 0:
            return np.array([], dtype=np.float32)

        probs = self.model.predict_proba(X_mat)
        # Class 1 probability
        return probs[:, 1].astype(np.float32)

    def save(self, file_path: str | Path) -> Path:
        """
        Serializes model artifact and feature configuration to disk.

        Args:
            file_path: Destination path.

        Returns:
            Saved file path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "params": self.params,
                "feature_columns": self.feature_columns,
                "best_iteration_": self.best_iteration_,
                "feature_importances_": self.feature_importances_,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> EntityMatcherModel:
        """
        Loads a serialized EntityMatcherModel from disk.

        Args:
            file_path: Path to serialized artifact.

        Returns:
            Reconstituted EntityMatcherModel.
        """
        state = joblib.load(file_path)
        instance = cls(feature_columns=state["feature_columns"])
        instance.params = state["params"]
        instance.model = state["model"]
        instance.best_iteration_ = state.get("best_iteration_")
        instance.feature_importances_ = state.get("feature_importances_")
        return instance


# ---------------------------------------------------------------------------
# Entity-disjoint splitting
# ---------------------------------------------------------------------------

def split_entity_disjoint(
    df: pl.DataFrame,
    s1_id_col: str = "source1_entity_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Splits candidate dataset into train and validation sets grouped strictly by Source1 entity ID
    to guarantee zero data leakage across splits.

    Args:
        df: Polars DataFrame of candidate pairs.
        s1_id_col: Column name of Source1 entity ID.
        test_size: Fraction of entities allocated to validation.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, val_df).
    """
    unique_s1 = df[s1_id_col].unique().to_numpy()
    np.random.seed(random_state)
    n_val = int(len(unique_s1) * test_size)
    shuffled = np.random.permutation(unique_s1)
    val_s1_set = set(shuffled[:n_val])

    train_df = df.filter(~pl.col(s1_id_col).is_in(val_s1_set))
    val_df = df.filter(pl.col(s1_id_col).is_in(val_s1_set))
    return train_df, val_df


# ---------------------------------------------------------------------------
# Convenience orchestrator
# ---------------------------------------------------------------------------

def train_matching_model(
    train_features: pl.DataFrame,
    val_features: Optional[pl.DataFrame] = None,
    label_col: str = "label",
    s1_id_col: str = "source1_entity_id",
    model_params: Optional[dict[str, Any]] = None,
    save_path: Optional[str | Path] = None,
) -> tuple[EntityMatcherModel, dict[str, Any]]:
    """
    Convenience orchestrator for training and validating the EntityMatcherModel.

    Args:
        train_features: Training DataFrame containing features and label.
        val_features: Optional validation DataFrame.
        label_col: Binary label column name.
        s1_id_col: Source1 ID column.
        model_params: Custom hyperparameter dictionary.
        save_path: Optional path to save model.

    Returns:
        Tuple of (fitted EntityMatcherModel, training_metrics_dict).
    """
    if val_features is None:
        train_df, val_df = split_entity_disjoint(train_features, s1_id_col=s1_id_col, test_size=0.2)
    else:
        train_df, val_df = train_features, val_features

    y_train = train_df[label_col].to_numpy()
    y_val = val_df[label_col].to_numpy() if val_df is not None and label_col in val_df.columns else None

    params = model_params or {}
    model = EntityMatcherModel(**params)
    model.fit(train_df, y_train, X_val=val_df, y_val=y_val)

    if save_path:
        model.save(save_path)

    metrics = {
        "train_samples": train_df.height,
        "val_samples": val_df.height if val_df is not None else 0,
        "best_iteration": model.best_iteration_,
        "feature_importances": model.feature_importances_,
    }
    return model, metrics


# =====================================================================
# End-to-end training script (executed via __main__)
# =====================================================================

def _run_full_training_pipeline() -> None:
    """
    Complete training pipeline:
        1. Load training sources + ground truth via DuckDB (10 k S1 slice)
        2. Normalize text, 4-route blocking candidate generation
        3. Schema standardization, attribute hydration, ground-truth labeling
        4. Entity-disjoint 80/20 train/val split
        5. 29-feature extraction (RapidFuzz + metadata)
        6. LightGBM training with early stopping
        7. Macro F0.5 threshold tuning on validation set
        8. Save model artifact + config to experiments/models/
    """
    # ---- lazy imports (only needed for the script path) ----
    import duckdb
    import pandas as pd

    from code.business_entity_resolution.src.preprocessing.normalize_name import normalize_business_name
    from code.business_entity_resolution.src.preprocessing.normalize_address import normalize_business_address
    from code.business_entity_resolution.src.blocking.exact_blocking import generate_exact_blocks
    from code.business_entity_resolution.src.blocking.token_blocking import generate_token_blocks
    from code.business_entity_resolution.src.blocking.address_blocking import generate_address_blocks
    from code.business_entity_resolution.src.blocking.ngram_blocking import generate_ngram_blocks
    from code.business_entity_resolution.src.features.pair_features import (
        FEATURE_COLUMNS as FEAT_COLS,
        standardize_candidate_schema,
        hydrate_candidate_pairs,
        construct_training_candidates,
        extract_pair_features,
    )
    from code.business_entity_resolution.src.evaluation.threshold_tuning import tune_f05_threshold
    from code.business_entity_resolution.src.evaluation.evaluate import evaluate_predictions

    tracemalloc.start()
    total_start = time.time()

    # ------------------------------------------------------------------ paths
    base_dir = Path(__file__).resolve().parents[4]
    train_dir = base_dir / "student_resource" / "dataset" / "train"
    s1_path = (train_dir / "train_source1.tsv").as_posix()
    s2_path = (train_dir / "train_source2.tsv").as_posix()
    s3_path = (train_dir / "train_source3.tsv").as_posix()
    gt_path = (train_dir / "train_ground_truth.tsv").as_posix()

    models_dir = Path(__file__).resolve().parents[2] / "experiments" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================ STEP 1
    print("=" * 70, flush=True)
    print("STEP 1: Slicing 10,000 S1 Entities & Constructing Target Pool", flush=True)
    print("=" * 70, flush=True)

    con = duckdb.connect()

    con.execute(f"""
        CREATE VIEW s1_raw AS SELECT * FROM read_csv_auto('{s1_path}', delim='\\t');
        CREATE VIEW gt_raw AS SELECT * FROM read_csv_auto('{gt_path}', delim='\\t');
        CREATE TABLE sample_s1 AS SELECT * FROM s1_raw LIMIT 10000;
        CREATE TABLE sample_gt AS
            SELECT g.* FROM gt_raw g
            JOIN sample_s1 s ON g.source1_entity_id = s.entity_id;
    """)

    s1_df_raw = con.execute("SELECT * FROM sample_s1").df()
    gt_df_raw = con.execute("SELECT * FROM sample_gt").df()

    # Build ground-truth mapping
    gt_map: dict[str, set[str]] = {}
    all_true_target_ids: set[str] = set()
    total_true_links = 0
    singleton_count = 0

    for _, row in gt_df_raw.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = row.get("matched_entity_ids")
        if pd.isna(m_str) or not str(m_str).strip() or str(m_str).lower() == "null":
            gt_map[s1] = set()
            singleton_count += 1
        else:
            ids = {x.strip() for x in str(m_str).split(",") if x.strip()}
            gt_map[s1] = ids
            total_true_links += len(ids)
            all_true_target_ids.update(ids)

    print(f"Total S1 entities: {len(s1_df_raw):,}", flush=True)
    print(f"Total true ground-truth positive links: {total_true_links:,}", flush=True)
    print(f"Total true singletons: {singleton_count:,} ({singleton_count/len(s1_df_raw)*100:.2f}%)", flush=True)

    # Build target pool: ensure all ground-truth targets are present + extra for negatives
    con.register("true_target_ids_df", pd.DataFrame({"entity_id": list(all_true_target_ids)}))

    con.execute(f"""
        CREATE VIEW s2_raw AS SELECT * FROM read_csv_auto('{s2_path}', delim='\\t');
        CREATE TABLE sample_s2 AS
            SELECT * FROM s2_raw WHERE entity_id IN (SELECT entity_id FROM true_target_ids_df)
            UNION
            SELECT * FROM (SELECT * FROM s2_raw LIMIT 100000);
    """)
    s2_df_raw = con.execute("SELECT * FROM sample_s2").df()

    con.execute(f"""
        CREATE VIEW s3_raw AS SELECT * FROM read_csv_auto('{s3_path}', delim='\\t');
        CREATE TABLE sample_s3 AS
            SELECT * FROM s3_raw WHERE entity_id IN (SELECT entity_id FROM true_target_ids_df)
            UNION
            SELECT * FROM (SELECT * FROM s3_raw LIMIT 100000);
    """)
    s3_df_raw = con.execute("SELECT * FROM sample_s3").df()
    con.close()

    print(f"Loaded target pools: Source 2 = {len(s2_df_raw):,} rows, Source 3 = {len(s3_df_raw):,} rows", flush=True)

    # ============================================================ STEP 2
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Normalization & 4-Route Candidate Generation", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    s1_df_raw["name_normalized"] = s1_df_raw["business_name"].apply(normalize_business_name)
    s1_df_raw["address_normalized"] = s1_df_raw["business_address"].apply(normalize_business_address)
    s2_df_raw["name_normalized"] = s2_df_raw["business_name"].apply(normalize_business_name)
    s2_df_raw["address_normalized"] = s2_df_raw["business_address"].apply(normalize_business_address)
    s3_df_raw["name_normalized"] = s3_df_raw["business_name"].apply(normalize_business_name)
    s3_df_raw["address_normalized"] = s3_df_raw["business_address"].apply(normalize_business_address)

    s1_prep = s1_df_raw.rename(columns={"entity_id": "s1_id"})[["s1_id", "country", "name_normalized", "address_normalized"]]
    s2_prep = s2_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]
    s3_prep = s3_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]

    # Route 1: Exact name match
    exact_s2 = generate_exact_blocks(s1_prep, s2_prep, key_column="name_normalized")
    exact_s3 = generate_exact_blocks(s1_prep, s3_prep, key_column="name_normalized")
    exact_s2["source"] = "S2"
    exact_s3["source"] = "S3"
    exact_df = pd.concat([exact_s2, exact_s3], ignore_index=True).drop_duplicates()
    exact_df["block_type"] = "exact"

    # Route 2: Rare-token blocking
    token_s2 = generate_token_blocks(s1_prep, s2_prep, max_token_frequency=100, min_token_length=3)
    token_s3 = generate_token_blocks(s1_prep, s3_prep, max_token_frequency=100, min_token_length=3)
    token_s2["source"] = "S2"
    token_s3["source"] = "S3"
    token_df = pd.concat([token_s2, token_s3], ignore_index=True).drop_duplicates()
    token_df["block_type"] = "token"

    # Route 3: Address-token blocking
    addr_s2 = generate_address_blocks(s1_prep, s2_prep, max_token_frequency=75, min_token_length=3)
    addr_s3 = generate_address_blocks(s1_prep, s3_prep, max_token_frequency=75, min_token_length=3)
    addr_s2["source"] = "S2"
    addr_s3["source"] = "S3"
    addr_df = pd.concat([addr_s2, addr_s3], ignore_index=True).drop_duplicates()
    addr_df["block_type"] = "address"

    # Route 4: Character n-gram blocking
    ngram_s2 = generate_ngram_blocks(s1_prep, s2_prep, ngram_size=3, max_ngram_frequency=50)
    ngram_s3 = generate_ngram_blocks(s1_prep, s3_prep, ngram_size=3, max_ngram_frequency=50)
    ngram_s2["source"] = "S2"
    ngram_s3["source"] = "S3"
    ngram_df = pd.concat([ngram_s2, ngram_s3], ignore_index=True).drop_duplicates()
    ngram_df["block_type"] = "ngram"

    all_raw_cands = pd.concat([exact_df, token_df, addr_df, ngram_df], ignore_index=True)

    four_route_candidates = (
        all_raw_cands.groupby(["s1_id", "matched_id", "source"], as_index=False)
        .agg(
            block_type=("block_type", lambda x: ",".join(sorted(set(x)))),
            number_of_blocks_hit=("block_type", "nunique"),
        )
    )

    gen_time = time.time() - t0
    print(f"Generated {len(four_route_candidates):,} unique candidate pairs in {gen_time:.2f}s", flush=True)

    # Candidate recall before ML
    union_cand_map: dict[str, set[str]] = defaultdict(set)
    for sid, mid in zip(four_route_candidates["s1_id"].values, four_route_candidates["matched_id"].values):
        union_cand_map[sid].add(mid)

    captured_true_links = sum(len(v & union_cand_map.get(k, set())) for k, v in gt_map.items())
    cand_recall_pct = (captured_true_links / total_true_links * 100) if total_true_links > 0 else 100.0
    print(f"\n>>> CANDIDATE RECALL BEFORE ML: {captured_true_links:,} / {total_true_links:,} ({cand_recall_pct:.2f}%) <<<", flush=True)

    # ============================================================ STEP 3
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Schema Standardization, Attribute Hydration & Ground-Truth Labeling", flush=True)
    print("=" * 70, flush=True)

    std_cand_df = standardize_candidate_schema(four_route_candidates)

    s1_polars = pl.from_pandas(s1_df_raw)
    s2_polars = pl.from_pandas(s2_df_raw)
    s3_polars = pl.from_pandas(s3_df_raw)

    hydrated_df = hydrate_candidate_pairs(
        std_cand_df,
        source1_table=s1_polars,
        source2_table=s2_polars,
        source3_table=s3_polars,
    )

    labeled_df = construct_training_candidates(hydrated_df, ground_truth=gt_map)

    pos_count = int(labeled_df["label"].sum())
    neg_count = labeled_df.height - pos_count
    pos_rate_pct = (pos_count / labeled_df.height * 100) if labeled_df.height > 0 else 0.0

    print(f"Total Hydrated Candidate Pairs: {labeled_df.height:,}", flush=True)
    print(f"Positive Pairs (label=1): {pos_count:,}", flush=True)
    print(f"Negative Pairs (label=0, Hard Negatives): {neg_count:,}", flush=True)
    print(f"Positive Rate: {pos_rate_pct:.2f}% (Class imbalance: 1 : {neg_count/max(pos_count, 1):.1f})", flush=True)

    # ============================================================ STEP 4
    print("\n" + "=" * 70, flush=True)
    print("STEP 4: Entity-Disjoint Train/Validation Split (80% Train / 20% Val)", flush=True)
    print("=" * 70, flush=True)

    train_cands, val_cands = split_entity_disjoint(
        labeled_df,
        s1_id_col="source1_entity_id",
        test_size=0.20,
        random_state=42,
    )

    train_s1_count = train_cands["source1_entity_id"].n_unique()
    val_s1_count = val_cands["source1_entity_id"].n_unique()

    train_s1_set = set(train_cands["source1_entity_id"].unique().to_list())
    val_s1_set = set(val_cands["source1_entity_id"].unique().to_list())
    overlap = len(train_s1_set & val_s1_set)
    assert overlap == 0, "FATAL: Entity leakage detected between train and validation!"

    print(f"Train Source1 Entities: {train_s1_count:,} ({train_cands.height:,} candidate pairs, {int(train_cands['label'].sum()):,} positives)", flush=True)
    print(f"Validation Source1 Entities: {val_s1_count:,} ({val_cands.height:,} candidate pairs, {int(val_cands['label'].sum()):,} positives)", flush=True)
    print(f"Zero Leakage Check: PASSED (Train/Val entity intersection = {overlap})", flush=True)

    # ============================================================ STEP 5
    print("\n" + "=" * 70, flush=True)
    print("STEP 5: Vectorized 29-Feature Extraction (RapidFuzz & Metadata)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    train_feats = extract_pair_features(train_cands)
    val_feats = extract_pair_features(val_cands)
    feat_time = time.time() - t0

    print(
        f"Extracted {len(FEAT_COLS)} pairwise features across "
        f"{train_feats.height + val_feats.height:,} pairs in {feat_time:.2f}s "
        f"({train_feats.height / max(feat_time, 0.01):,.0f} pairs/sec)",
        flush=True,
    )

    # ============================================================ STEP 6
    print("\n" + "=" * 70, flush=True)
    print("STEP 6: LightGBM Pairwise Classifier Training", flush=True)
    print("=" * 70, flush=True)

    y_train = train_feats["label"].to_numpy()
    y_val = val_feats["label"].to_numpy()

    model = EntityMatcherModel(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        random_state=42,
        early_stopping_rounds=30,
    )

    t0 = time.time()
    model.fit(train_feats, y_train, X_val=val_feats, y_val=y_val)
    train_time = time.time() - t0

    print(f"LightGBM trained successfully in {train_time:.2f}s (Best iteration: {model.best_iteration_})", flush=True)

    # Top Feature Importances
    print("\nTop 10 Feature Importances:", flush=True)
    sorted_imps = sorted(model.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in sorted_imps:
        print(f"  {feat:30s}: {imp:.1f}", flush=True)

    # ============================================================ STEP 7
    print("\n" + "=" * 70, flush=True)
    print("STEP 7: Validation Probability Prediction & Threshold Grid Tuning (Macro F0.5)", flush=True)
    print("=" * 70, flush=True)

    val_probs = model.predict_proba(val_feats)
    val_scored = val_feats.with_columns(pl.Series("match_probability", val_probs))

    val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_s1_set}

    tuning_res = tune_f05_threshold(
        val_scored_candidates=val_scored,
        val_ground_truth=val_gt_map,
        threshold_min=0.10,
        threshold_max=0.90,
        threshold_step=0.05,
        all_s1_ids=list(val_s1_set),
    )

    best_threshold = tuning_res["best_threshold"]
    best_macro_f05 = tuning_res["best_macro_f05"]
    best_m = tuning_res["best_metrics"]

    print("\nThreshold Scan Results (Macro F0.5 Targeting):", flush=True)
    print(
        f"{'Threshold':>10s} | {'Macro F0.5':>10s} | {'Precision':>10s} | "
        f"{'Recall':>10s} | {'Singleton Acc':>14s} | {'Avg Pred Matches':>16s}",
        flush=True,
    )
    print("-" * 85, flush=True)
    for rec in tuning_res["tuning_history"]:
        marker = " <== OPTIMAL" if rec["threshold"] == best_threshold else ""
        print(
            f"{rec['threshold']:10.2f} | {rec['macro_f05']:10.4f} | "
            f"{rec['macro_precision']:10.4f} | {rec['macro_recall']:10.4f} | "
            f"{rec['singleton_accuracy']:13.2%} | {rec['avg_predicted_matches']:16.2f}{marker}",
            flush=True,
        )

    # ============================================================ STEP 8
    print("\n" + "=" * 70, flush=True)
    print("STEP 8: Final Model Evaluation at Optimal Threshold", flush=True)
    print("=" * 70, flush=True)

    print(f"Optimal Decision Threshold:       {best_threshold:.2f}", flush=True)
    print(f"Entity-Level Macro F0.5:           {best_macro_f05:.4f}", flush=True)
    print(f"Macro Precision:                   {best_m['macro_precision']:.4f}", flush=True)
    print(f"Macro Recall (relative to GT):     {best_m['macro_recall']:.4f}", flush=True)
    print(f"Singleton / No-Match Accuracy:     {best_m['singleton_accuracy']:.2%}", flush=True)
    print(f"Exact Match Set Rate:              {best_m['exact_set_match_rate']:.2%}", flush=True)
    print(f"Average True Matches per S1:       {best_m['avg_true_matches']:.2f}", flush=True)
    print(f"Average Predicted Matches per S1:  {best_m['avg_predicted_matches']:.2f}", flush=True)

    # Pair-level validation metrics
    val_pred_mask = val_probs >= best_threshold
    pair_tp = int(np.sum((val_pred_mask == 1) & (y_val == 1)))
    pair_fp = int(np.sum((val_pred_mask == 1) & (y_val == 0)))
    pair_fn = int(np.sum((val_pred_mask == 0) & (y_val == 1)))
    pair_precision = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) > 0 else 0.0
    pair_recall = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) > 0 else 0.0
    pair_f05 = (
        (1.25 * pair_precision * pair_recall) / (0.25 * pair_precision + pair_recall)
        if (0.25 * pair_precision + pair_recall) > 0
        else 0.0
    )

    print(f"\nPair-Level Validation Metrics (at T={best_threshold}):", flush=True)
    print(f"  Pair Precision: {pair_precision:.4f} (TP={pair_tp:,}, FP={pair_fp:,})", flush=True)
    print(f"  Pair Recall:    {pair_recall:.4f} (FN={pair_fn:,})", flush=True)
    print(f"  Pair F0.5:      {pair_f05:.4f}", flush=True)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.time() - total_start

    print(f"\nResource Profiling:", flush=True)
    print(f"  Total Runtime:  {total_elapsed:.2f}s", flush=True)
    print(f"  Peak Memory:    {peak_mem / (1024 * 1024):.2f} MB ({peak_mem / (1024 * 1024 * 1024):.2f} GB)", flush=True)

    # ============================================================ STEP 9
    print("\n" + "=" * 70, flush=True)
    print("STEP 9: Saving Model Artifact & Threshold Config", flush=True)
    print("=" * 70, flush=True)

    model_save_path = models_dir / "entity_matcher_lgbm.joblib"
    config_save_path = models_dir / "entity_matcher_threshold_config.json"

    model.save(model_save_path)

    config_data = {
        "model_type": "LightGBM_EntityMatcherModel",
        "feature_count": len(FEAT_COLS),
        "feature_columns": FEAT_COLS,
        "best_threshold": best_threshold,
        "best_iteration": model.best_iteration_,
        "validation_metrics": {
            "macro_f05": best_m.get("macro_f05"),
            "macro_precision": best_m.get("macro_precision"),
            "macro_recall": best_m.get("macro_recall"),
            "singleton_accuracy": best_m.get("singleton_accuracy"),
            "exact_set_match_rate": best_m.get("exact_set_match_rate"),
            "avg_predicted_matches": best_m.get("avg_predicted_matches"),
            "avg_true_matches": best_m.get("avg_true_matches"),
            "total_entities": best_m.get("total_entities"),
            "singleton_entities": best_m.get("singleton_entities"),
        },
        "pair_level_metrics": {
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "pair_f05": pair_f05,
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
        },
        "candidate_recall_before_ml": cand_recall_pct,
        "total_candidate_pairs": labeled_df.height,
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "positive_rate_pct": pos_rate_pct,
        "train_s1_entities": train_s1_count,
        "val_s1_entities": val_s1_count,
        "feature_importances": model.feature_importances_,
        "total_runtime_seconds": total_elapsed,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }

    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    print(f"Model saved to:         {model_save_path}", flush=True)
    print(f"Threshold config saved: {config_save_path}", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_full_training_pipeline()
