"""
End-to-end test inference and submission artifact generator.

Takes ranked candidates, extracts pairwise features, executes batch model scoring,
applies the calibrated Macro F0.5 decision threshold, and generates formatted TSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence
import polars as pl

from ..ranking.candidate_ranker import CandidateRanker
from ..features.pair_features import extract_batch_features
from ..models.train_model import EntityMatcherModel
from ..models.predict import predict_matches_batch, format_matching_results


def run_candidate_ranking_and_inference(
    candidate_pairs_df: pl.DataFrame,
    model_path: str | Path,
    all_test_s1_ids: Sequence[str],
    output_dir: str | Path = "output",
    threshold: float = 0.5,
    top_k_candidates: int = 30,
) -> tuple[Path, Path]:
    """
    Executes candidate ranking, feature extraction, model scoring, and submission TSV generation.

    Args:
        candidate_pairs_df: DataFrame of raw test candidate pairs from blocking.
        model_path: Path to serialized trained LightGBM model.
        all_test_s1_ids: Complete collection of all test Source1 entity IDs.
        output_dir: Folder to write matching_results.tsv and candidate_pairs.tsv.
        threshold: Decision probability threshold.
        top_k_candidates: Maximum candidates to keep per S1 in candidate_pairs.tsv.

    Returns:
        Tuple of (matching_results_tsv_path, candidate_pairs_tsv_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rank & truncate candidates
    ranker = CandidateRanker(top_k=top_k_candidates)
    ranked_cands = ranker.rank_and_truncate(candidate_pairs_df)

    # 2. Export output/candidate_pairs.tsv
    cand_pairs_path = out_dir / "candidate_pairs.tsv"
    cand_tsv_df = ranker.format_candidate_pairs_tsv(ranked_cands, all_s1_ids=all_test_s1_ids)
    cand_tsv_df.write_csv(cand_pairs_path, separator="\t")

    # 3. Extract pairwise features for ranked candidates
    feat_df = extract_batch_features(ranked_cands)

    # 4. Load model and predict matches
    model = EntityMatcherModel.load(model_path)
    accepted_matches = predict_matches_batch(
        model=model,
        candidate_df=feat_df,
        threshold=threshold,
    )

    # 5. Export output/matching_results.tsv
    matching_results_path = out_dir / "matching_results.tsv"
    format_matching_results(
        accepted_matches=accepted_matches,
        all_s1_ids=all_test_s1_ids,
        output_path=matching_results_path,
    )

    return matching_results_path, cand_pairs_path


def run_inference_pipeline(
    test_dir: str | Path = "dataset/test",
    model_path: str | Path = "code/business_entity_resolution/experiments/models/4route_lgbm.joblib",
    output_dir: str | Path = "output",
    config: Optional[dict[str, Any]] = None,
) -> tuple[Path, Path]:
    """
    CLI wrapper for end-to-end inference pipeline.
    """
    from code.business_entity_resolution.src.models.predict import run_batch_inference
    print(f"Executing inference pipeline on {test_dir}...", flush=True)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_path = out_dir / "matching_results.tsv"
    c_path = out_dir / "candidate_pairs.tsv"
    if not m_path.exists() or not c_path.exists():
        run_batch_inference(
            model_path=str(model_path),
            test_dir=str(test_dir),
            output_dir=str(output_dir),
        )
    return m_path, c_path


def main():
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Inference Pipeline.")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset folder.")
    parser.add_argument("--model-path", default="code/business_entity_resolution/experiments/models/4route_lgbm.joblib", help="Path to trained model.")
    parser.add_argument("--output-dir", default="output", help="Path to output directory.")
    args = parser.parse_args()

    print("Running inference pipeline...")
    run_inference_pipeline(args.test_dir, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
