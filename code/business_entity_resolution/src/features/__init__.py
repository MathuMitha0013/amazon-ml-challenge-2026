"""
Pairwise feature extraction, schema adapter, and attribute hydration for machine learning.
"""

from .pair_features import (
    FEATURE_COLUMNS,
    extract_single_pair_features,
    extract_pair_features,
    extract_batch_features,
    standardize_candidate_schema,
    hydrate_candidate_pairs,
    construct_training_candidates,
)

__all__ = [
    "FEATURE_COLUMNS",
    "extract_single_pair_features",
    "extract_pair_features",
    "extract_batch_features",
    "standardize_candidate_schema",
    "hydrate_candidate_pairs",
    "construct_training_candidates",
]
