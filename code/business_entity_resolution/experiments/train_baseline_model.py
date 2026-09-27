"""
Baseline LightGBM Entity Matching Training & Validation on Realistic Candidate Pool.

Uses:
- Person 2 candidate generation (Exact + Rare-Token UNION on 10k S1 slice)
- 28 RapidFuzz & structural pairwise features
- Entity-disjoint 80/20 train/validation split
- Hard-negative mining from blocking candidates
- Macro F0.5 threshold optimization and singleton preservation
"""

import os
import sys
import json
import time
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from unidecode import unidecode

from code.business_entity_resolution.src.features.pair_features import (
    FEATURE_COLUMNS,
    standardize_candidate_schema,
    hydrate_candidate_pairs,
    construct_training_candidates,
    extract_pair_features,
)
from code.business_entity_resolution.src.models.train_model import EntityMatcherModel, split_entity_disjoint
from code.business_entity_resolution.src.evaluation.threshold_tuning import tune_f05_threshold
from code.business_entity_resolution.src.evaluation.evaluate import evaluate_predictions, compute_candidate_statistics
from code.business_entity_resolution.src.evaluation.candidate_recall import evaluate_candidate_recall


# --- Person 2's Normalization & Blocking Functions ---

def normalize_business_name_p2(raw_name: str) -> str:
    if raw_name is None:
        return ""
    name = str(raw_name).strip()
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = unidecode(name)
    name = name.lower()
    name = re.sub(r"\bl\s*\.?\s*l\s*\.?\s*c\.?\b", "llc", name)
    name = re.sub(r"\bp\s*\.?\s*l\s*\.?\s*l\s*\.?\s*c\.?\b", "pllc", name)
    name = re.sub(r"\bi\s*\.?\s*n\s*\.?\s*c\.?\b", "inc", name)
    name = re.sub(r"\bl\s*\.?\s*t\s*\.?\s*d\.?\b", "ltd", name)
    name = re.sub(r"\bp\s*\.?\s*c\.?\b", "pc", name)
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def generate_exact_blocks_p2(source1_data: pd.DataFrame, source2_3_data: pd.DataFrame, key_column: str = "name_normalized") -> pd.DataFrame:
    candidates = source1_data.merge(
        source2_3_data,
        on=["country", key_column],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )
    res = candidates[["s1_id", "matched_id"]].drop_duplicates()
    res["block_type"] = "exact"
    return res


def generate_token_blocks_p2(source1_data: pd.DataFrame, source2_3_data: pd.DataFrame, max_token_frequency: int = 100, min_token_length: int = 3) -> pd.DataFrame:
    s1 = source1_data[["s1_id", "country", "name_normalized"]].copy()
    target = source2_3_data[["matched_id", "country", "name_normalized"]].copy()

    s1["token"] = s1["name_normalized"].fillna("").str.split()
    s1 = s1.explode("token")

    target["token"] = target["name_normalized"].fillna("").str.split()
    target = target.explode("token")

    s1 = s1[s1["token"].str.len() >= min_token_length]
    target = target[target["token"].str.len() >= min_token_length]

    token_frequency = (
        target.groupby(["country", "token"])
        .size()
        .reset_index(name="frequency")
    )
    token_frequency = token_frequency[token_frequency["frequency"] <= max_token_frequency]

    s1 = s1.merge(token_frequency, on=["country", "token"], how="inner")
    target = target.merge(token_frequency, on=["country", "token"], how="inner")

    candidates = s1.merge(
        target,
        on=["country", "token"],
        how="inner",
        suffixes=("_s1", "_target"),
    )

    res = candidates[["s1_id", "matched_id"]].drop_duplicates()
    res["block_type"] = "token"
    return res


def run_baseline_training():
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "student_resource" / "dataset" / "train"
    if not train_dir.exists():
        train_dir = base_dir / "student_resource" / "dataset" / "dataset" / "train"
    s1_path = (train_dir / "train_source1.tsv").as_posix()
    s2_path = (train_dir / "train_source2.tsv").as_posix()
    s3_path = (train_dir / "train_source3.tsv").as_posix()
    gt_path = (train_dir / "train_ground_truth.tsv").as_posix()

    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Slicing 10,000 S1 Entities & Constructing Target Pool")
    print("=" * 70)

    con = duckdb.connect()

    sample_s1_query = f"""
        CREATE VIEW s1_raw AS SELECT * FROM read_csv_auto('{s1_path}', delim='\\t');
        CREATE VIEW gt_raw AS SELECT * FROM read_csv_auto('{gt_path}', delim='\\t');
        CREATE TABLE sample_s1 AS 
            SELECT * FROM s1_raw LIMIT 10000;
        CREATE TABLE sample_gt AS 
            SELECT g.* FROM gt_raw g
            JOIN sample_s1 s ON g.source1_entity_id = s.entity_id;
    """
    con.execute(sample_s1_query)

    s1_df_raw = con.execute("SELECT * FROM sample_s1").df()
    gt_df_raw = con.execute("SELECT * FROM sample_gt").df()

    # Ground truth mapping
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

    print(f"Total S1 entities: {len(s1_df_raw):,}")
    print(f"Total true ground-truth positive links: {total_true_links:,}")
    print(f"Total true singletons: {singleton_count:,} ({singleton_count/len(s1_df_raw)*100:.2f}%)")

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

    print(f"Loaded target pools: Source 2 = {len(s2_df_raw):,} rows, Source 3 = {len(s3_df_raw):,} rows")

    print("\n=" * 70)
    print("STEP 2: Preprocessing & Candidate Generation (Person 2 Routes)")
    print("=" * 70)

    t0 = time.time()
    s1_df_raw["name_normalized"] = s1_df_raw["business_name"].apply(normalize_business_name_p2)
    s2_df_raw["name_normalized"] = s2_df_raw["business_name"].apply(normalize_business_name_p2)
    s3_df_raw["name_normalized"] = s3_df_raw["business_name"].apply(normalize_business_name_p2)

    s1_prep = s1_df_raw.rename(columns={"entity_id": "s1_id"})[["s1_id", "country", "name_normalized"]]
    s2_prep = s2_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized"]]
    s3_prep = s3_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized"]]

    exact_s2 = generate_exact_blocks_p2(s1_prep, s2_prep)
    exact_s3 = generate_exact_blocks_p2(s1_prep, s3_prep)
    token_s2 = generate_token_blocks_p2(s1_prep, s2_prep, max_token_frequency=100, min_token_length=3)
    token_s3 = generate_token_blocks_p2(s1_prep, s3_prep, max_token_frequency=100, min_token_length=3)

    exact_s2["source"] = "S2"
    exact_s3["source"] = "S3"
    token_s2["source"] = "S2"
    token_s3["source"] = "S3"

    all_raw_cands = pd.concat([exact_s2, exact_s3, token_s2, token_s3], ignore_index=True)

    # Multi-route grouping matching Person 2's candidate_generation.py schema
    person2_candidates = (
        all_raw_cands.groupby(["s1_id", "matched_id", "source"], as_index=False)
        .agg(
            block_type=("block_type", lambda x: ",".join(sorted(set(x)))),
            number_of_blocks_hit=("block_type", "nunique"),
        )
    )

    gen_time = time.time() - t0
    print(f"Generated {len(person2_candidates):,} unique candidate pairs in {gen_time:.2f}s")

    # Step 2b: Measure candidate recall before ML
    union_cand_map = defaultdict(set)
    for sid, mid in zip(person2_candidates["s1_id"].values, person2_candidates["matched_id"].values):
        union_cand_map[sid].add(mid)

    captured_true_links = sum(len(v & union_cand_map.get(k, set())) for k, v in gt_map.items())
    cand_recall_pct = (captured_true_links / total_true_links * 100) if total_true_links > 0 else 100.0

    print(f"\n>>> CANDIDATE RECALL BEFORE ML: {captured_true_links:,} / {total_true_links:,} ({cand_recall_pct:.2f}%) <<<")

    print("\n=" * 70)
    print("STEP 3: Adapter Standardisation, Attribute Hydration & Ground-Truth Labeling")
    print("=" * 70)

    # 1. Standardize schema: s1_id -> source1_entity_id, matched_id -> candidate_entity_id
    std_cand_df = standardize_candidate_schema(person2_candidates)

    # 2. Hydrate text attributes from source tables
    s1_polars = pl.from_pandas(s1_df_raw)
    s2_polars = pl.from_pandas(s2_df_raw)
    s3_polars = pl.from_pandas(s3_df_raw)

    hydrated_df = hydrate_candidate_pairs(
        std_cand_df,
        source1_table=s1_polars,
        source2_table=s2_polars,
        source3_table=s3_polars,
    )

    # 3. Label candidate pairs using ground truth (label=1 for true matches, 0 for blocked hard negatives)
    labeled_df = construct_training_candidates(hydrated_df, ground_truth=gt_map)

    pos_count = int(labeled_df["label"].sum())
    neg_count = labeled_df.height - pos_count
    pos_rate_pct = (pos_count / labeled_df.height * 100) if labeled_df.height > 0 else 0.0

    print(f"Total Hydrated Candidate Pairs: {labeled_df.height:,}")
    print(f"Positive Pairs (label=1): {pos_count:,}")
    print(f"Negative Pairs (label=0, Hard Negatives): {neg_count:,}")
    print(f"Positive Rate: {pos_rate_pct:.2f}% (Class imbalance: 1 : {neg_count/pos_count:.1f})")

    print("\n=" * 70)
    print("STEP 4: Entity-Disjoint Train/Validation Split (80% Train / 20% Val)")
    print("=" * 70)

    train_cands, val_cands = split_entity_disjoint(
        labeled_df,
        s1_id_col="source1_entity_id",
        test_size=0.20,
        random_state=42,
    )

    train_s1_count = train_cands["source1_entity_id"].n_unique()
    val_s1_count = val_cands["source1_entity_id"].n_unique()

    # Verify zero leakage
    train_s1_set = set(train_cands["source1_entity_id"].unique().to_list())
    val_s1_set = set(val_cands["source1_entity_id"].unique().to_list())
    overlap = len(train_s1_set & val_s1_set)
    assert overlap == 0, "FATAL: Entity leakage detected between train and validation!"

    print(f"Train Source1 Entities: {train_s1_count:,} ({train_cands.height:,} candidate pairs, {int(train_cands['label'].sum()):,} positives)")
    print(f"Validation Source1 Entities: {val_s1_count:,} ({val_cands.height:,} candidate pairs, {int(val_cands['label'].sum()):,} positives)")
    print(f"Zero Leakage Check: PASSED (Train/Val entity intersection = {overlap})")

    print("\n=" * 70)
    print("STEP 5: Vectorized 28-Feature Extraction (RapidFuzz & Metadata)")
    print("=" * 70)

    t0 = time.time()
    train_feats = extract_pair_features(train_cands)
    val_feats = extract_pair_features(val_cands)
    feat_time = time.time() - t0

    print(f"Extracted {len(FEATURE_COLUMNS)} pairwise features across {train_feats.height + val_feats.height:,} pairs in {feat_time:.2f}s ({train_feats.height / feat_time:,.0f} pairs/sec)")

    print("\n=" * 70)
    print("STEP 6: LightGBM Pairwise Classifier Training")
    print("=" * 70)

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

    print(f"LightGBM trained successfully in {train_time:.2f}s (Best iteration: {model.best_iteration_})")

    # Top 8 most important features
    print("\nTop 8 Feature Importances:")
    sorted_imps = sorted(model.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:8]
    for feat, imp in sorted_imps:
        print(f"  {feat:30s}: {imp:.1f}")

    print("\n=" * 70)
    print("STEP 7: Validation Probability Prediction & Threshold Grid Tuning (Macro F0.5)")
    print("=" * 70)

    val_probs = model.predict_proba(val_feats)
    val_scored = val_feats.with_columns(pl.Series("match_probability", val_probs))

    # Ground truth sub-mapping for validation S1 entities
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

    print("\nThreshold Scan Results (Macro F0.5 Targeting):")
    print(f"{'Threshold':>10s} | {'Macro F0.5':>10s} | {'Precision':>10s} | {'Recall':>10s} | {'Singleton Acc':>14s} | {'Avg Pred Matches':>16s}")
    print("-" * 85)
    for rec in tuning_res["tuning_history"]:
        marker = " <== OPTIMAL" if rec["threshold"] == best_threshold else ""
        print(f"{rec['threshold']:10.2f} | {rec['macro_f05']:10.4f} | {rec['macro_precision']:10.4f} | {rec['macro_recall']:10.4f} | {rec['singleton_accuracy']:13.2%} | {rec['avg_predicted_matches']:16.2f}{marker}")

    print("\n" + "=" * 70)
    print("STEP 8: Final Model Evaluation at Optimal Threshold")
    print("=" * 70)
    print(f"Optimal Decision Threshold:       {best_threshold:.2f}")
    print(f"Entity-Level Macro F0.5:           {best_macro_f05:.4f}")
    print(f"Macro Precision:                   {best_m['macro_precision']:.4f}")
    print(f"Macro Recall (relative to GT):     {best_m['macro_recall']:.4f}")
    print(f"Singleton / No-Match Accuracy:     {best_m['singleton_accuracy']:.2%}")
    print(f"Exact Match Set Rate:              {best_m['exact_set_match_rate']:.2%}")
    print(f"Average True Matches per S1:       {best_m['avg_true_matches']:.2f}")
    print(f"Average Predicted Matches per S1:  {best_m['avg_predicted_matches']:.2f}")

    # Pair-level validation metrics at best threshold
    val_pred_mask = val_probs >= best_threshold
    pair_tp = int(np.sum((val_pred_mask == 1) & (y_val == 1)))
    pair_fp = int(np.sum((val_pred_mask == 1) & (y_val == 0)))
    pair_fn = int(np.sum((val_pred_mask == 0) & (y_val == 1)))
    pair_precision = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) > 0 else 0.0
    pair_recall = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) > 0 else 0.0
    pair_f05 = ((1.25 * pair_precision * pair_recall) / (0.25 * pair_precision + pair_recall)) if (0.25 * pair_precision + pair_recall) > 0 else 0.0

    print(f"\nPair-Level Validation Metrics (at T={best_threshold}):")
    print(f"  Pair Precision: {pair_precision:.4f} (TP={pair_tp:,}, FP={pair_fp:,})")
    print(f"  Pair Recall:    {pair_recall:.4f} (FN={pair_fn:,})")
    print(f"  Pair F0.5:      {pair_f05:.4f}")

    print("\n=" * 70)
    print("STEP 9: Saving Model Artifact & Threshold Config")
    print("=" * 70)

    model_save_path = models_dir / "baseline_lgbm.joblib"
    config_save_path = models_dir / "baseline_threshold_config.json"

    model.save(model_save_path)

    config_data = {
        "model_type": "LightGBM_EntityMatcherModel",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "best_threshold": best_threshold,
        "best_iteration": model.best_iteration_,
        "validation_metrics": best_m,
        "pair_level_metrics": {
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "pair_f05": pair_f05,
            "tp": pair_tp,
            "fp": pair_fp,
            "fn": pair_fn,
        },
        "candidate_recall_before_ml": cand_recall_pct,
        "train_s1_entities": train_s1_count,
        "val_s1_entities": val_s1_count,
    }

    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"Model saved to:         {model_save_path}")
    print(f"Threshold config saved: {config_save_path}")


if __name__ == "__main__":
    run_baseline_training()
