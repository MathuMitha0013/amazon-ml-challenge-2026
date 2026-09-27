"""
Machine learning training, hard-negative mining, and batch prediction modules.
"""

from .train_model import (
    EntityMatcherModel,
    train_matching_model,
    split_entity_disjoint,
)
from .predict import (
    predict_matches_batch,
    format_matching_results,
)

__all__ = [
    "EntityMatcherModel",
    "train_matching_model",
    "split_entity_disjoint",
    "predict_matches_batch",
    "format_matching_results",
]
