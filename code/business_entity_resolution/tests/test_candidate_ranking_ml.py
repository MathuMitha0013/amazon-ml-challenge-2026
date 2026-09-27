"""
Comprehensive Synthetic Test Suite for Candidate Ranking & ML Matching modules.

Tests cover:
- Exact match
- Typos & abbreviations
- Address component variations
- Same name / wrong address (hard negative)
- Same address / wrong name (hard negative)
- Singleton / no-match entities
- Multi-target matches (S2 and S3)
- Unicode & transliterations
- Model training, saving, loading, probability prediction
- Threshold tuning and entity-level Macro F0.5 evaluation
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import polars as pl

from src.ranking.candidate_ranker import CandidateRanker, rank_candidates
from src.features.pair_features import (
    FEATURE_COLUMNS,
    extract_single_pair_features,
    extract_pair_features,
)
from src.models.train_model import (
    EntityMatcherModel,
    train_matching_model,
    split_entity_disjoint,
)
from src.models.predict import predict_matches_batch, format_matching_results
from src.evaluation.evaluate import (
    compute_single_entity_metrics,
    evaluate_predictions,
    compute_candidate_statistics,
)
from src.evaluation.threshold_tuning import tune_f05_threshold
from src.evaluation.candidate_recall import evaluate_candidate_recall
from src.evaluation.ground_truth_analysis import analyze_ground_truth


@pytest.fixture
def synthetic_records_dataset():
    """
    Constructs a rich synthetic dataset testing all realistic noise patterns and edge cases.
    """
    rows = [
        # 1. Exact match (S1 -> S2)
        {
            "source1_entity_id": "S1-1001",
            "candidate_entity_id": "S2-5001",
            "s1_business_name": "Acme Global Logistics Inc",
            "candidate_business_name": "Acme Global Logistics Inc",
            "s1_business_address": "123 Main Street, Suite 400, New York, NY 10001",
            "candidate_business_address": "123 Main Street, Suite 400, New York, NY 10001",
            "s1_country": "US",
            "candidate_country": "US",
            "number_of_blocks_hit": 4,
            "label": 1,
        },
        # 2. Name Typo + Abbreviation (S1 -> S3)
        {
            "source1_entity_id": "S1-1002",
            "candidate_entity_id": "S3-8002",
            "s1_business_name": "Apex Engineering & Technologies Corp",
            "candidate_business_name": "Apex Engg and Tech Corporation",
            "s1_business_address": "456 Silicon Valley Blvd, San Jose, CA 95134",
            "candidate_business_address": "456 Silicon Vly Blvd, Ste 200, San Jose, CA 95134",
            "s1_country": "US",
            "candidate_country": "US",
            "number_of_blocks_hit": 3,
            "label": 1,
        },
        # 3. Hard Negative: Same Name, completely wrong Address & Country
        {
            "source1_entity_id": "S1-1002",
            "candidate_entity_id": "S2-5009",
            "s1_business_name": "Apex Engineering & Technologies Corp",
            "candidate_business_name": "Apex Engineering Corp",
            "s1_business_address": "456 Silicon Valley Blvd, San Jose, CA 95134",
            "candidate_business_address": "789 Industrial Road, Sector 18, Gurgaon, 122015",
            "s1_country": "US",
            "candidate_country": "IN",
            "number_of_blocks_hit": 1,
            "label": 0,
        },
        # 4. Hard Negative: Same Address, completely wrong Name
        {
            "source1_entity_id": "S1-1001",
            "candidate_entity_id": "S3-8099",
            "s1_business_name": "Acme Global Logistics Inc",
            "candidate_business_name": "Blue Ocean Seafood Restaurant LLC",
            "s1_business_address": "123 Main Street, Suite 400, New York, NY 10001",
            "candidate_business_address": "123 Main Street, Suite 400, New York, NY 10001",
            "s1_country": "US",
            "candidate_country": "US",
            "number_of_blocks_hit": 1,
            "label": 0,
        },
        # 5. Multiple matches (S1-1003 matches both S2-5003 and S3-8003)
        {
            "source1_entity_id": "S1-1003",
            "candidate_entity_id": "S2-5003",
            "s1_business_name": "Bharat Agro Foods Private Limited",
            "candidate_business_name": "Bharat Agro Foods Pvt Ltd",
            "s1_business_address": "Plot 24, MIDC Industrial Area, Andheri East, Mumbai 400093",
            "candidate_business_address": "Plot No 24 MIDC Ind Area, Andheri E, Mumbai, MH 400093",
            "s1_country": "IN",
            "candidate_country": "IN",
            "number_of_blocks_hit": 4,
            "label": 1,
        },
        {
            "source1_entity_id": "S1-1003",
            "candidate_entity_id": "S3-8003",
            "s1_business_name": "Bharat Agro Foods Private Limited",
            "candidate_business_name": "Bharat Agrofoods",
            "s1_business_address": "Plot 24, MIDC Industrial Area, Andheri East, Mumbai 400093",
            "candidate_business_address": "24 MIDC, Mumbai 400093",
            "s1_country": "IN",
            "candidate_country": "IN",
            "number_of_blocks_hit": 2,
            "label": 1,
        },
        # 6. Unicode / Accented Characters (France distribution)
        {
            "source1_entity_id": "S1-1004",
            "candidate_entity_id": "S2-5004",
            "s1_business_name": "Café de Paris Société Anonyme",
            "candidate_business_name": "Cafe de Paris SA",
            "s1_business_address": "15 Rue de la Paix, 75002 Paris",
            "candidate_business_address": "15 RUE DE LA PAIX, PARIS 75002",
            "s1_country": "FR",
            "candidate_country": "FR",
            "number_of_blocks_hit": 3,
            "label": 1,
        },
        # 7. Low-similarity False Candidate for S1-1004
        {
            "source1_entity_id": "S1-1004",
            "candidate_entity_id": "S3-8007",
            "s1_business_name": "Café de Paris Société Anonyme",
            "candidate_business_name": "Boulangerie Patisserie Parisienne",
            "s1_business_address": "15 Rue de la Paix, 75002 Paris",
            "candidate_business_address": "88 Boulevard Saint-Germain, 75005 Paris",
            "s1_country": "FR",
            "candidate_country": "FR",
            "number_of_blocks_hit": 1,
            "label": 0,
        },
        # 8. True Singleton with noisy negative candidate (S1-1005 is a singleton)
        {
            "source1_entity_id": "S1-1005",
            "candidate_entity_id": "S2-5088",
            "s1_business_name": "Quantum Precision Consulting Ltd",
            "candidate_business_name": "Precision Dynamics LLC",
            "s1_business_address": "100 Innovation Way, Austin, TX 78701",
            "candidate_business_address": "500 Commerce St, Dallas, TX 75201",
            "s1_country": "US",
            "candidate_country": "US",
            "number_of_blocks_hit": 1,
            "label": 0,
        },
    ]
    return pl.DataFrame(rows)


def test_candidate_ranker_scoring_and_truncation(synthetic_records_dataset):
    """Test CandidateRanker generates scores, ranks, and truncates candidates per S1."""
    ranker = CandidateRanker(top_k=2, min_score=0.0)
    scored = ranker.compute_ranking_signals(synthetic_records_dataset)

    assert "rank_score" in scored.columns
    assert "name_ratio" in scored.columns
    assert "country_match" in scored.columns

    # Exact match should have a near 1.0 rank score
    exact_row = scored.filter(pl.col("source1_entity_id") == "S1-1001")
    assert exact_row["rank_score"][0] > 0.85

    # Truncate to top-1
    top1_df = ranker.rank_and_truncate(synthetic_records_dataset, top_k=1)
    # Each S1 in result should have at most 1 candidate
    counts = top1_df.group_by("source1_entity_id").agg(pl.count("candidate_entity_id"))
    assert (counts["candidate_entity_id"] <= 1).all()

    # Verify format_candidate_pairs_tsv
    all_s1 = ["S1-1001", "S1-1002", "S1-1003", "S1-1004", "S1-1005", "S1-1006_missing"]
    cand_tsv_df = ranker.format_candidate_pairs_tsv(top1_df, all_s1_ids=all_s1)
    assert "source1_entity_id" in cand_tsv_df.columns
    assert "candidate_entity_ids" in cand_tsv_df.columns
    assert cand_tsv_df.height == len(all_s1)


def test_pairwise_feature_extraction(synthetic_records_dataset):
    """Test feature engineering extracts all expected features with valid values."""
    feat_df = extract_pair_features(synthetic_records_dataset)

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns
        vals = feat_df[col].to_numpy()
        assert not np.isnan(vals).any(), f"NaN found in feature column {col}"
        assert not np.isinf(vals).any(), f"Inf found in feature column {col}"

    # Verify specific feature behavior
    # Exact match row
    exact_feats = feat_df.filter(pl.col("source1_entity_id") == "S1-1001")
    assert exact_feats["name_exact_match"][0] == 1.0
    assert exact_feats["addr_exact_match"][0] == 1.0
    assert exact_feats["country_match"][0] == 1.0
    assert exact_feats["is_s2"][0] == 1.0

    # Unicode row
    uni_feats = feat_df.filter(pl.col("source1_entity_id") == "S1-1004").filter(pl.col("candidate_entity_id") == "S2-5004")
    assert uni_feats["name_partial_ratio"][0] > 0.85
    assert uni_feats["name_wratio"][0] > 0.80
    assert uni_feats["addr_token_jaccard"][0] > 0.50


def test_entity_matcher_model_train_predict(synthetic_records_dataset):
    """Test LightGBM EntityMatcherModel fitting, prediction, save and load."""
    feat_df = extract_pair_features(synthetic_records_dataset)

    # Train model
    model = EntityMatcherModel(n_estimators=20, learning_rate=0.1, max_depth=3)
    y_train = feat_df["label"].to_numpy()
    model.fit(feat_df, y_train)

    # Predict probabilities
    probs = model.predict_proba(feat_df)
    assert len(probs) == feat_df.height
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Save and reload
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"
        model.save(model_path)
        assert model_path.exists()

        loaded_model = EntityMatcherModel.load(model_path)
        loaded_probs = loaded_model.predict_proba(feat_df)
        np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_predict_and_format_matching_results(synthetic_records_dataset):
    """Test predict_matches_batch and format_matching_results."""
    feat_df = extract_pair_features(synthetic_records_dataset)
    model = EntityMatcherModel(n_estimators=20, learning_rate=0.1, max_depth=3)
    model.fit(feat_df, feat_df["label"].to_numpy())

    accepted = predict_matches_batch(model, feat_df, threshold=0.3)
    assert "source1_entity_id" in accepted.columns
    assert "candidate_entity_id" in accepted.columns
    assert "match_probability" in accepted.columns

    all_s1 = ["S1-1001", "S1-1002", "S1-1003", "S1-1004", "S1-1005"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_tsv = Path(tmp_dir) / "matching_results.tsv"
        formatted_df = format_matching_results(accepted, all_s1_ids=all_s1, output_path=out_tsv)

        assert out_tsv.exists()
        assert formatted_df.height == len(all_s1)
        assert list(formatted_df.columns) == ["source1_entity_id", "matched_entity_ids"]

        # Read back TSV to verify tab separation
        content = out_tsv.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == len(all_s1) + 1  # header + rows
        assert "\t" in lines[0]


def test_entity_level_evaluate():
    """Test entity-level Macro F0.5 calculation, singletons, and exact match rates."""
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),  # true singleton
        "S1-3": {"S2-30"},
    }

    # Perfect prediction
    pred_perfect = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),
        "S1-3": {"S2-30"},
    }
    res = evaluate_predictions(gt, pred_perfect, beta=0.5)
    assert res["macro_f05"] == 1.0
    assert res["singleton_accuracy"] == 1.0
    assert res["exact_set_match_rate"] == 1.0

    # Prediction with false merge on singleton
    pred_bad_singleton = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-99"},  # false merge on true singleton -> penalty
        "S1-3": {"S2-30"},
    }
    res_bad = evaluate_predictions(gt, pred_bad_singleton, beta=0.5)
    assert res_bad["macro_f05"] < 1.0
    assert res_bad["singleton_accuracy"] == 0.0

    # Single entity metric edge cases
    # True singleton correct:
    m1 = compute_single_entity_metrics(set(), set(), beta=0.5)
    assert m1["f_score"] == 1.0 and m1["is_singleton"] == 1.0

    # True singleton false positive:
    m2 = compute_single_entity_metrics(set(), {"S2-1"}, beta=0.5)
    assert m2["f_score"] == 0.0 and m2["singleton_correct"] == 0.0


def test_threshold_tuning(synthetic_records_dataset):
    """Test threshold optimizer explores cutoffs and finds optimal Macro F0.5."""
    feat_df = extract_pair_features(synthetic_records_dataset)
    model = EntityMatcherModel(n_estimators=30, learning_rate=0.1, max_depth=3)
    model.fit(feat_df, feat_df["label"].to_numpy())

    probs = model.predict_proba(feat_df)
    scored_val = feat_df.with_columns(pl.Series("match_probability", probs))

    # Ground truth mapping
    gt_map = {
        "S1-1001": {"S2-5001"},
        "S1-1002": {"S3-8002"},
        "S1-1003": {"S2-5003", "S3-8003"},
        "S1-1004": {"S2-5004"},
        "S1-1005": set(),  # singleton
    }

    all_s1 = list(gt_map.keys())
    tuning_res = tune_f05_threshold(
        scored_val,
        val_ground_truth=gt_map,
        threshold_min=0.1,
        threshold_max=0.9,
        threshold_step=0.1,
        all_s1_ids=all_s1,
    )

    assert "best_threshold" in tuning_res
    assert "best_macro_f05" in tuning_res
    assert len(tuning_res["tuning_history"]) > 0
    assert tuning_res["best_macro_f05"] > 0.0


def test_candidate_recall_and_gt_analysis():
    """Test candidate recall metrics and ground truth profiling."""
    gt = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": set(),
        "S1-3": {"S2-30"},
    }
    cands = {
        "S1-1": {"S2-10", "S3-20", "S2-99"},
        "S1-2": set(),
        "S1-3": {"S2-30"},
    }

    # Candidate recall check (all 3 true pairs captured)
    recall_stats = evaluate_candidate_recall(gt, cands, total_s1_count=3, total_target_count=100)
    assert recall_stats["candidate_recall"] == 1.0
    assert recall_stats["true_matches_captured"] == 3
    assert "reduction_ratio_pct" in recall_stats

    # Ground truth analysis check
    gt_profile = analyze_ground_truth(gt)
    assert gt_profile["total_entities"] == 3
    assert gt_profile["singleton_count"] == 1
    assert gt_profile["s2_matches"] == 2
    assert gt_profile["s3_matches"] == 1


def test_end_to_end_synthetic_pipeline(synthetic_records_dataset):
    """Test full training + candidate ranking + inference + output TSV generation on synthetic data."""
    from src.pipeline.train_pipeline import train_and_calibrate_pipeline
    from src.pipeline.inference_pipeline import run_candidate_ranking_and_inference

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        out_dir = tmp_path / "output"

        # Ground truth
        gt_map = {
            "S1-1001": {"S2-5001"},
            "S1-1002": {"S3-8002"},
            "S1-1003": {"S2-5003", "S3-8003"},
            "S1-1004": {"S2-5004"},
            "S1-1005": set(),
        }

        # 1. Train and calibrate
        train_res = train_and_calibrate_pipeline(
            candidate_pairs_df=synthetic_records_dataset,
            ground_truth=gt_map,
            output_model_dir=models_dir,
            top_k_candidates=5,
            model_params={"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
            val_split_size=0.3,
        )
        assert Path(train_res["model_path"]).exists()
        assert Path(train_res["config_path"]).exists()

        # 2. Run inference
        all_s1 = ["S1-1001", "S1-1002", "S1-1003", "S1-1004", "S1-1005"]
        match_tsv, cand_tsv = run_candidate_ranking_and_inference(
            candidate_pairs_df=synthetic_records_dataset,
            model_path=train_res["model_path"],
            all_test_s1_ids=all_s1,
            output_dir=out_dir,
            threshold=train_res["best_threshold"],
            top_k_candidates=5,
        )

        assert match_tsv.exists()
        assert cand_tsv.exists()

        # 3. Verify format of exported TSVs
        match_df = pl.read_csv(match_tsv, separator="\t")
        cand_df = pl.read_csv(cand_tsv, separator="\t")

        assert list(match_df.columns) == ["source1_entity_id", "matched_entity_ids"]
        assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
        assert match_df.height == len(all_s1)
        assert cand_df.height == len(all_s1)


def test_person2_adapter_hydration_and_labeling():
    """Test adapting Person 2 candidate output schema (s1_id, matched_id, block_type), hydrating attributes, and labeling."""
    from src.features.pair_features import (
        standardize_candidate_schema,
        hydrate_candidate_pairs,
        construct_training_candidates,
    )

    # 1. Person 2 candidate output schema
    person2_output = pl.DataFrame(
        {
            "s1_id": ["S1-100", "S1-100", "S1-200", "S1-300"],
            "matched_id": ["S2-500", "S3-800", "S2-501", "S3-999"],
            "source": ["S2", "S3", "S2", "S3"],
            "block_type": ["exact", "token", "exact,token", "exact"],
            "number_of_blocks_hit": [1, 1, 2, 1],
        }
    )

    # Standardize
    std_df = standardize_candidate_schema(person2_output)
    assert "source1_entity_id" in std_df.columns
    assert "candidate_entity_id" in std_df.columns
    assert "block_type" in std_df.columns
    assert "number_of_blocks_hit" in std_df.columns

    # 2. Source tables
    s1_table = pl.DataFrame(
        {
            "entity_id": ["S1-100", "S1-200", "S1-300"],
            "business_name": ["Apex Logistics Inc", "Bharat Foods Pvt Ltd", "Café Paris"],
            "business_address": ["123 Main St, New York", "Plot 4 MIDC, Mumbai", "15 Rue Paris"],
            "country": ["US", "IN", "FR"],
        }
    )
    s2_table = pl.DataFrame(
        {
            "entity_id": ["S2-500", "S2-501"],
            "business_name": ["Apex Logistics", "Bharat Foods Limited"],
            "business_address": ["123 Main Street, NY", "Plot 4 MIDC, Mumbai"],
            "country": ["US", "IN"],
        }
    )
    s3_table = pl.DataFrame(
        {
            "entity_id": ["S3-800", "S3-999"],
            "business_name": ["Apex Log", "Completely Wrong Bakery"],
            "business_address": ["123 Main St", "999 Random Blvd, Texas"],
            "country": ["US", "US"],
        }
    )

    # Hydrate
    hydrated = hydrate_candidate_pairs(
        std_df,
        source1_table=s1_table,
        source2_table=s2_table,
        source3_table=s3_table,
    )
    assert hydrated.height == 4
    assert "s1_business_name" in hydrated.columns
    assert "candidate_business_name" in hydrated.columns
    assert "s1_business_address" in hydrated.columns
    assert "candidate_business_address" in hydrated.columns
    assert "s1_country" in hydrated.columns
    assert "candidate_country" in hydrated.columns

    # 3. Labeling with ground truth
    gt = {
        "S1-100": {"S2-500", "S3-800"},  # multi-match
        "S1-200": {"S2-501"},           # single match
        "S1-300": set(),                 # singleton: S3-999 is a hard negative!
    }

    labeled = construct_training_candidates(hydrated, ground_truth=gt)
    assert "label" in labeled.columns
    labels = labeled["label"].to_list()
    assert labels == [1, 1, 1, 0]  # First 3 true matches, last 1 hard negative (labeled 0)

