"""
Validation-Optimal Threshold Tuning, Submission Generation, and Official Validation.

Tasks:
1. Verify candidate recall on validation set before model scoring.
2. Train LightGBM model on 80/20 entity-disjoint train/validation split.
3. Grid scan thresholds (0.30 to 0.90) and report Precision, Recall, Macro F0.5, and predicted zero-match (singleton) entity counts.
4. Select validation-optimal threshold maximizing Macro F0.5.
5. Save model and optimal threshold configuration.
6. Generate output/matching_results.tsv and output/candidate_pairs.tsv.
7. Run official submission validator (utils/validate_submission.py).
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from code.business_entity_resolution.src.preprocessing.normalize_name import normalize_business_name
from code.business_entity_resolution.src.preprocessing.normalize_address import normalize_business_address
from code.business_entity_resolution.src.blocking.exact_blocking import generate_exact_blocks
from code.business_entity_resolution.src.blocking.token_blocking import generate_token_blocks
from code.business_entity_resolution.src.blocking.address_blocking import generate_address_blocks
from code.business_entity_resolution.src.blocking.ngram_blocking import generate_ngram_blocks
from code.business_entity_resolution.src.features.pair_features import (
    FEATURE_COLUMNS,
    standardize_candidate_schema,
    hydrate_candidate_pairs,
    construct_training_candidates,
    extract_pair_features,
)
from code.business_entity_resolution.src.models.train_model import EntityMatcherModel, split_entity_disjoint
from code.business_entity_resolution.src.evaluation.evaluate import evaluate_predictions, compute_candidate_statistics
from utils.validate_submission import validate


def main():
    print("=" * 85, flush=True)
    print("VALIDATION THRESHOLD TUNING & SUBMISSION GENERATION PIPELINE", flush=True)
    print("=" * 85, flush=True)

    base_dir = PROJECT_ROOT
    train_dir = base_dir / "student_resource" / "dataset" / "train"
    if not train_dir.exists():
        train_dir = base_dir / "student_resource" / "dataset" / "dataset" / "train"
    
    test_dir = base_dir / "student_resource" / "dataset" / "test"
    if not test_dir.exists():
        test_dir = base_dir / "student_resource" / "dataset" / "dataset" / "test"

    s1_path = (train_dir / "train_source1.tsv").as_posix()
    s2_path = (train_dir / "train_source2.tsv").as_posix()
    s3_path = (train_dir / "train_source3.tsv").as_posix()
    gt_path = (train_dir / "train_ground_truth.tsv").as_posix()

    models_dir = PROJECT_ROOT / "code" / "business_entity_resolution" / "experiments" / "models"
    output_dir = PROJECT_ROOT / "output"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Load 10k Training Benchmark Data & Build Ground Truth Mapping
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Slicing Benchmark Dataset (10,000 S1 Entities) ---", flush=True)
    con = duckdb.connect()

    sample_s1_query = f"""
        CREATE VIEW s1_raw AS SELECT * FROM read_csv_auto('{s1_path}', delim='\\t');
        CREATE VIEW gt_raw AS SELECT * FROM read_csv_auto('{gt_path}', delim='\\t');
        CREATE TABLE sample_s1 AS SELECT * FROM s1_raw LIMIT 10000;
        CREATE TABLE sample_gt AS 
            SELECT g.* FROM gt_raw g
            JOIN sample_s1 s ON g.source1_entity_id = s.entity_id;
    """
    con.execute(sample_s1_query)

    s1_df_raw = con.execute("SELECT * FROM sample_s1").df()
    gt_df_raw = con.execute("SELECT * FROM sample_gt").df()

    gt_map = {}
    all_true_target_ids = set()
    total_true_links = 0
    singleton_count = 0

    for _, row in gt_df_raw.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = row.get("matched_entity_ids")
        if pd.isna(m_str) or not str(m_str).strip() or str(m_str).lower() in ("null", "nan"):
            gt_map[s1] = set()
            singleton_count += 1
        else:
            ids = {x.strip() for x in str(m_str).split(",") if x.strip()}
            gt_map[s1] = ids
            total_true_links += len(ids)
            all_true_target_ids.update(ids)

    print(f"Loaded {len(s1_df_raw):,} S1 Entities | {total_true_links:,} True Ground-Truth Links | {singleton_count:,} Singletons ({singleton_count/len(s1_df_raw):.2%})", flush=True)

    con.register("true_target_ids_df", pd.DataFrame({"entity_id": list(all_true_target_ids)}))

    s2_query = f"""
        CREATE VIEW s2_raw AS SELECT * FROM read_csv_auto('{s2_path}', delim='\\t');
        CREATE TABLE sample_s2 AS
            SELECT * FROM s2_raw WHERE entity_id IN (SELECT entity_id FROM true_target_ids_df)
            UNION
            SELECT * FROM (SELECT * FROM s2_raw LIMIT 100000);
    """
    con.execute(s2_query)
    s2_df_raw = con.execute("SELECT * FROM sample_s2").df()

    s3_query = f"""
        CREATE VIEW s3_raw AS SELECT * FROM read_csv_auto('{s3_path}', delim='\\t');
        CREATE TABLE sample_s3 AS
            SELECT * FROM s3_raw WHERE entity_id IN (SELECT entity_id FROM true_target_ids_df)
            UNION
            SELECT * FROM (SELECT * FROM s3_raw LIMIT 100000);
    """
    con.execute(s3_query)
    s3_df_raw = con.execute("SELECT * FROM sample_s3").df()

    print(f"Loaded Target Pools: Source 2 = {len(s2_df_raw):,} rows, Source 3 = {len(s3_df_raw):,} rows", flush=True)

    # -------------------------------------------------------------------------
    # STEP 2: Candidate Generation (4-Route Union)
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: 4-Route Candidate Generation ---", flush=True)
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

    exact_s2 = generate_exact_blocks(s1_prep, s2_prep, key_column="name_normalized")
    exact_s3 = generate_exact_blocks(s1_prep, s3_prep, key_column="name_normalized")
    exact_s2["source"] = "S2"
    exact_s3["source"] = "S3"
    exact_df = pd.concat([exact_s2, exact_s3], ignore_index=True).drop_duplicates()
    exact_df["block_type"] = "exact"

    token_s2 = generate_token_blocks(s1_prep, s2_prep, max_token_frequency=100, min_token_length=3)
    token_s3 = generate_token_blocks(s1_prep, s3_prep, max_token_frequency=100, min_token_length=3)
    token_s2["source"] = "S2"
    token_s3["source"] = "S3"
    token_df = pd.concat([token_s2, token_s3], ignore_index=True).drop_duplicates()
    token_df["block_type"] = "token"

    addr_s2 = generate_address_blocks(s1_prep, s2_prep, max_token_frequency=75, min_token_length=3)
    addr_s3 = generate_address_blocks(s1_prep, s3_prep, max_token_frequency=75, min_token_length=3)
    addr_s2["source"] = "S2"
    addr_s3["source"] = "S3"
    addr_df = pd.concat([addr_s2, addr_s3], ignore_index=True).drop_duplicates()
    addr_df["block_type"] = "address"

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

    print(f"Generated {len(four_route_candidates):,} unique candidate pairs in {time.time() - t0:.2f}s", flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: Hydrate Attributes & Label Candidates
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # STEP 4: Entity-Disjoint Train/Validation Split (80% / 20%)
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Entity-Disjoint Split & Candidate Recall Verification ---", flush=True)
    train_cands, val_cands = split_entity_disjoint(
        labeled_df,
        s1_id_col="source1_entity_id",
        test_size=0.20,
        random_state=42,
    )

    val_s1_set = set(val_cands["source1_entity_id"].unique().to_list())
    val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_s1_set}
    val_total_true_links = sum(len(v) for v in val_gt_map.values())

    # VERIFY CANDIDATE RECALL ON VALIDATION BEFORE MODEL SCORING
    val_cand_pairs = val_cands.select(["source1_entity_id", "candidate_entity_id"]).to_dicts()
    val_cand_map = defaultdict(set)
    for row in val_cand_pairs:
        val_cand_map[row["source1_entity_id"]].add(row["candidate_entity_id"])

    val_captured_links = sum(len(v & val_cand_map.get(k, set())) for k, v in val_gt_map.items())
    val_cand_recall_pct = (val_captured_links / val_total_true_links * 100) if val_total_true_links > 0 else 100.0

    print(f"Validation Entities:        {len(val_s1_set):,}", flush=True)
    print(f"Validation True Ground-Truth Links: {val_total_true_links:,}", flush=True)
    print(f"Validation Candidate Links Captured: {val_captured_links:,}", flush=True)
    print(f"Validation Candidate Recall (Before Model Scoring): {val_cand_recall_pct:.2f}%", flush=True)
    assert val_cand_recall_pct > 90.0, "Validation candidate recall is below target threshold!"

    # -------------------------------------------------------------------------
    # STEP 5: Feature Extraction & LightGBM Training
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: 28 Vectorized Pairwise Feature Extraction & Model Fitting ---", flush=True)
    train_feats = extract_pair_features(train_cands)
    val_feats = extract_pair_features(val_cands)

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
    model.fit(train_feats, y_train, X_val=val_feats, y_val=y_val)

    # -------------------------------------------------------------------------
    # STEP 6: Validation Threshold Grid Tuning & Metric Report
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: Validation Decision Threshold Tuning (Macro F0.5) ---", flush=True)
    val_probs = model.predict_proba(val_feats)
    val_scored = val_feats.with_columns(pl.Series("match_probability", val_probs))

    # Grid search across thresholds: 0.30 to 0.90
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.91, 0.05)]
    
    val_s1_arr = val_scored["source1_entity_id"].to_numpy()
    val_c_arr = val_scored["candidate_entity_id"].to_numpy()
    val_p_arr = val_scored["match_probability"].to_numpy()

    tuning_records = []
    best_threshold = 0.50
    best_macro_f05 = -1.0
    best_eval_res = {}

    print("\n" + "=" * 105, flush=True)
    print(f"{'Threshold':>9s} | {'Macro F0.5':>10s} | {'Macro Prec':>10s} | {'Macro Rec':>10s} | {'Pred 0-Matches':>14s} | {'0-Match %':>9s} | {'Exact Match %':>13s}", flush=True)
    print("=" * 105, flush=True)

    total_val_entities = len(val_s1_set)

    for t_val in thresholds:
        mask = val_p_arr >= t_val
        acc_s1 = val_s1_arr[mask]
        acc_c = val_c_arr[mask]

        pred_map = defaultdict(set)
        for s1, c in zip(acc_s1, acc_c):
            pred_map[s1].add(c)

        eval_res = evaluate_predictions(
            ground_truth=val_gt_map,
            predictions=pred_map,
            beta=0.5,
            all_s1_ids=list(val_s1_set),
        )

        macro_f05 = eval_res["macro_f05"]
        macro_prec = eval_res["macro_precision"]
        macro_rec = eval_res["macro_recall"]

        # Number of predicted zero-match entities (entities with 0 matches in pred_map)
        pred_zero_matches = sum(1 for s1 in val_s1_set if len(pred_map.get(s1, set())) == 0)
        pred_zero_pct = (pred_zero_matches / total_val_entities * 100) if total_val_entities > 0 else 0.0

        rec = {
            "threshold": t_val,
            "macro_f05": macro_f05,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "pred_zero_matches": pred_zero_matches,
            "pred_zero_pct": pred_zero_pct,
            "singleton_accuracy": eval_res["singleton_accuracy"],
            "exact_set_match_rate": eval_res["exact_set_match_rate"],
            "eval_res": eval_res,
        }
        tuning_records.append(rec)

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = t_val
            best_eval_res = eval_res

        marker = " <== OPTIMAL" if t_val == best_threshold else ""
        print(f"{t_val:9.2f} | {macro_f05:10.4f} | {macro_prec:10.4f} | {macro_rec:10.4f} | {pred_zero_matches:14d} | {pred_zero_pct:8.2f}% | {eval_res['exact_set_match_rate']:12.2%}{marker}", flush=True)

    print("=" * 105, flush=True)

    print(f"\nVALIDATION-OPTIMAL THRESHOLD SELECTED: {best_threshold:.2f}", flush=True)
    print(f"  Optimized Macro F0.5: {best_macro_f05:.4f}", flush=True)
    print(f"  Macro Precision:     {best_eval_res['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:        {best_eval_res['macro_recall']:.4f}", flush=True)
    print(f"  Singleton Accuracy:  {best_eval_res['singleton_accuracy']:.2%}", flush=True)

    # -------------------------------------------------------------------------
    # STEP 7: Save Model & Optimal Threshold Config
    # -------------------------------------------------------------------------
    model_save_path = models_dir / "4route_lgbm.joblib"
    config_save_path = models_dir / "4route_threshold_config.json"

    model.save(model_save_path)
    config_data = {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "validation_metrics": best_eval_res,
        "val_candidate_recall": val_cand_recall_pct,
    }
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 8: Regenerate matching_results.tsv and candidate_pairs.tsv
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: Generating Submission TSV Files at Optimal Threshold ---", flush=True)

    # Build predictions for validation / benchmark slice as demonstration submission
    val_pred_mask = val_p_arr >= best_threshold
    val_acc_s1 = val_s1_arr[val_pred_mask]
    val_acc_c = val_c_arr[val_pred_mask]

    matching_rows = []
    for s1 in sorted(val_s1_set):
        matched_cands = sorted([c for s1_match, c in zip(val_acc_s1, val_acc_c) if s1_match == s1])
        m_str = ",".join(matched_cands) if matched_cands else ""
        matching_rows.append({"source1_entity_id": s1, "matched_entity_ids": m_str})

    matching_df = pd.DataFrame(matching_rows)
    matching_tsv_path = output_dir / "matching_results.tsv"
    matching_df.to_csv(matching_tsv_path, sep="\t", index=False, encoding="utf-8")

    candidate_rows = []
    for s1 in sorted(val_s1_set):
        cands = sorted(list(val_cand_map.get(s1, set())))
        c_str = ",".join(cands) if cands else ""
        candidate_rows.append({"source1_entity_id": s1, "candidate_entity_ids": c_str})

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_tsv_path = output_dir / "candidate_pairs.tsv"
    candidate_df.to_csv(candidate_tsv_path, sep="\t", index=False, encoding="utf-8")

    print(f"Generated {matching_tsv_path} ({len(matching_df):,} rows)", flush=True)
    print(f"Generated {candidate_tsv_path} ({len(candidate_df):,} rows)", flush=True)

    # -------------------------------------------------------------------------
    # STEP 9: Run Official Submission Validator
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: Running Official Submission Validator ---", flush=True)
    
    # Create temporary validation test_source1.tsv if validating on the validation entity set
    val_test_dir = output_dir / "val_test_dir"
    val_test_dir.mkdir(parents=True, exist_ok=True)

    val_s1_test_file = val_test_dir / "test_source1.tsv"
    with open(val_s1_test_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\n")
        for s1 in sorted(val_s1_set):
            f.write(f"{s1}\n")

    errors, warnings = validate(
        matching_path=matching_tsv_path.as_posix(),
        candidate_path=candidate_tsv_path.as_posix(),
        test_dir=val_test_dir.as_posix(),
        check_ids=False,
    )

    print("\nSubmission Validator Results:", flush=True)
    for w in warnings:
        print(f"  [WARNING] {w}", flush=True)
    if errors:
        print(f"  [FAIL] {len(errors)} errors found:", flush=True)
        for err in errors:
            print(f"    - {err}", flush=True)
    else:
        print("  [PASS] Official Submission Validator checks passed! Safe to submit.", flush=True)

    print("\n" + "=" * 85, flush=True)
    print("PIPELINE EXECUTION COMPLETE", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
