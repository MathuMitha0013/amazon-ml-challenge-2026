"""
Competition evaluation metric: Macro-averaged F0.5 per Source1 entity.

Future Responsibility:
- Implement official competition scoring logic:
  F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Handle edge cases (empty ground truth & empty prediction -> score 1.0; empty prediction with non-empty GT -> 0.0).
- Compute precision, recall, and singleton classification accuracy.
"""

from typing import Any
from pathlib import Path


def compute_entity_f05(
    true_ids: set[str],
    pred_ids: set[str],
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """
    Computes Precision, Recall, and F_beta score for a single Source1 entity.

    Args:
        true_ids: Set of true matching S2/S3 IDs.
        pred_ids: Set of predicted matching S2/S3 IDs.
        beta: Weighting parameter (0.5 for precision emphasis).

    Returns:
        Tuple of (precision, recall, f05).
    """
    raise NotImplementedError("compute_entity_f05 will be implemented in subsequent phases.")


def compute_macro_f05(
    ground_truth: dict[str, set[str]] | str | Path,
    predictions: dict[str, set[str]] | str | Path,
) -> dict[str, float]:
    """
    Computes macro-averaged F0.5 across all Source1 entities in evaluation set.

    Args:
        ground_truth: Ground truth mapping or path to TSV.
        predictions: Predictions mapping or path to TSV.

    Returns:
        Dictionary containing:
        - macro_f05: Overall macro F0.5 score
        - macro_precision: Average precision
        - macro_recall: Average recall
        - singleton_accuracy: Accuracy on true zero-match entities
    """
    raise NotImplementedError("compute_macro_f05 will be implemented in subsequent phases.")
