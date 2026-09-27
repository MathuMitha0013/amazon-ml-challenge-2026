"""
Pairwise LightGBM matching model training, validation, and serialization.

Provides the EntityMatcherModel class with full support for entity-disjoint validation splits,
configurable hyperparameters, hard negative weighting, and early stopping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence
import joblib
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from ..features.pair_features import FEATURE_COLUMNS


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
