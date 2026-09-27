"""
Comprehensive Multi-Route Blocking Benchmark (10,000 Source 1 Entities).

Evaluates:
1. Exact normalized name blocking
2. Rare-token inverted index blocking
3. Address-based blocking (rare address tokens & numbers)
4. Character n-gram blocking (3-grams)
5. Multi-route UNION combination

Reports:
- Exact recall
- Token recall
- Address recall
- N-gram recall
- Combined recall
- Total candidate pairs
- Average candidates per S1
- P95 candidates
- P99 candidates
- Maximum candidates
- All-true-matches-captured percentage
"""

import time
import sys
from pathlib import Path
from collections import defaultdict
import duckdb
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

from code.business_entity_resolution.src.preprocessing.normalize_name import normalize_business_name
from code.business_entity_resolution.src.preprocessing.normalize_address import normalize_business_address
from code.business_entity_resolution.src.blocking.exact_blocking import generate_exact_blocks
from code.business_entity_resolution.src.blocking.token_blocking import generate_token_blocks
from code.business_entity_resolution.src.blocking.address_blocking import generate_address_blocks
from code.business_entity_resolution.src.blocking.ngram_blocking import generate_ngram_blocks
from code.business_entity_resolution.src.blocking.candidate_generation import combine_candidate_routes


def run_benchmark():
    base_dir = Path(__file__).resolve().parents[3]
    train_dir = base_dir / "student_resource" / "dataset" / "train"
    if not train_dir.exists():
        train_dir = base_dir / "student_resource" / "dataset" / "dataset" / "train"
    s1_path = (train_dir / "train_source1.tsv").as_posix()
    s2_path = (train_dir / "train_source2.tsv").as_posix()
    s3_path = (train_dir / "train_source3.tsv").as_posix()
    gt_path = (train_dir / "train_ground_truth.tsv").as_posix()

    print("=" * 70, flush=True)
    print("STEP 1: Slicing 10,000 S1 Entities & Constructing Target Pool", flush=True)
    print("=" * 70, flush=True)

    con = duckdb.connect()

    sample_s1_query = f"""
        CREATE VIEW s1_raw AS SELECT * FROM read_csv_auto('{s1_path}', delim='\\t');
        CREATE VIEW gt_raw AS SELECT * FROM read_csv_auto('{gt_path}', delim='\\t');
        CREATE TABLE sample_s1 AS 
            SELECT s.* FROM s1_raw s
            LIMIT 10000;
        CREATE TABLE sample_gt AS 
            SELECT g.* FROM gt_raw g
            JOIN sample_s1 s ON g.source1_entity_id = s.entity_id;
    """
    con.execute(sample_s1_query)

    s1_df_raw = con.execute("SELECT * FROM sample_s1").df()
    gt_df_raw = con.execute("SELECT * FROM sample_gt").df()

    print(f"Sample S1 entities: {len(s1_df_raw):,}", flush=True)
    print(f"Ground truth rows:  {len(gt_df_raw):,}", flush=True)

    gt_map = {}
    all_true_target_ids = set()
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

    print(f"Total True Positive Links in Ground Truth: {total_true_links:,}", flush=True)
    print(f"Singletons (0 matches) in slice:           {singleton_count:,} ({singleton_count/len(s1_df_raw)*100:.2f}%)", flush=True)

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

    print(f"Loaded target pools: Source 2 = {len(s2_df_raw):,} rows, Source 3 = {len(s3_df_raw):,} rows", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Normalizing Names & Addresses", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    s1_df_raw["name_normalized"] = s1_df_raw["business_name"].apply(normalize_business_name)
    s1_df_raw["address_normalized"] = s1_df_raw["business_address"].apply(normalize_business_address)

    s2_df_raw["name_normalized"] = s2_df_raw["business_name"].apply(normalize_business_name)
    s2_df_raw["address_normalized"] = s2_df_raw["business_address"].apply(normalize_business_address)

    s3_df_raw["name_normalized"] = s3_df_raw["business_name"].apply(normalize_business_name)
    s3_df_raw["address_normalized"] = s3_df_raw["business_address"].apply(normalize_business_address)
    print(f"Normalization completed in {time.time() - t0:.2f}s", flush=True)

    s1_prep = s1_df_raw.rename(columns={"entity_id": "s1_id"})[["s1_id", "country", "name_normalized", "address_normalized"]]
    s2_prep = s2_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]
    s3_prep = s3_df_raw.rename(columns={"entity_id": "matched_id"})[["matched_id", "country", "name_normalized", "address_normalized"]]

    # Route 1: Exact Name
    print("\nExecuting Route 1: Exact Name Blocking...", flush=True)
    t0 = time.time()
    exact_s2 = generate_exact_blocks(s1_prep, s2_prep, key_column="name_normalized")
    exact_s3 = generate_exact_blocks(s1_prep, s3_prep, key_column="name_normalized")
    exact_df = pd.concat([exact_s2, exact_s3], ignore_index=True).drop_duplicates()
    exact_time = time.time() - t0

    # Route 2: Rare Name Token
    print("Executing Route 2: Rare-Token Blocking...", flush=True)
    t0 = time.time()
    token_s2 = generate_token_blocks(s1_prep, s2_prep, max_token_frequency=100, min_token_length=3)
    token_s3 = generate_token_blocks(s1_prep, s3_prep, max_token_frequency=100, min_token_length=3)
    token_df = pd.concat([token_s2, token_s3], ignore_index=True).drop_duplicates()
    token_time = time.time() - t0

    # Route 3: Address-Based Blocking
    print("Executing Route 3: Address-Based Blocking...", flush=True)
    t0 = time.time()
    addr_s2 = generate_address_blocks(s1_prep, s2_prep, max_token_frequency=75, min_token_length=3)
    addr_s3 = generate_address_blocks(s1_prep, s3_prep, max_token_frequency=75, min_token_length=3)
    addr_df = pd.concat([addr_s2, addr_s3], ignore_index=True).drop_duplicates()
    addr_time = time.time() - t0

    # Route 4: Character N-Gram Blocking
    print("Executing Route 4: Character N-Gram Blocking...", flush=True)
    t0 = time.time()
    ngram_s2 = generate_ngram_blocks(s1_prep, s2_prep, ngram_size=3, max_ngram_frequency=50)
    ngram_s3 = generate_ngram_blocks(s1_prep, s3_prep, ngram_size=3, max_ngram_frequency=50)
    ngram_df = pd.concat([ngram_s2, ngram_s3], ignore_index=True).drop_duplicates()
    ngram_time = time.time() - t0

    # Vectorized fast evaluation helper
    def eval_route(df_pairs, route_name):
        cand_map = defaultdict(set)
        for sid, mid in zip(df_pairs["s1_id"].values, df_pairs["matched_id"].values):
            cand_map[sid].add(mid)
        cap = sum(len(v & cand_map.get(k, set())) for k, v in gt_map.items())
        rec = (cap / total_true_links * 100) if total_true_links > 0 else 100.0
        return len(df_pairs), cap, rec, cand_map

    exact_pairs, exact_cap, exact_rec, _ = eval_route(exact_df, "Exact Name")
    token_pairs, token_cap, token_rec, _ = eval_route(token_df, "Rare Token")
    addr_pairs, addr_cap, addr_rec, _ = eval_route(addr_df, "Address Token")
    ngram_pairs, ngram_cap, ngram_rec, _ = eval_route(ngram_df, "Character N-Gram")

    # Multi-Route UNION
    print("\nMerging 4 Routes into Multi-Route UNION...", flush=True)
    t0 = time.time()
    union_df = pd.concat([exact_df, token_df, addr_df, ngram_df], ignore_index=True).drop_duplicates(subset=["s1_id", "matched_id"])
    union_time = time.time() - t0

    union_pairs, union_cap, union_rec, union_cand_map = eval_route(union_df, "4-Route UNION")

    all_s1_ids = list(s1_df_raw["entity_id"])
    cands_per_s1 = [len(union_cand_map.get(s1, set())) for s1 in all_s1_ids]
    counts_arr = np.array(cands_per_s1)

    full_capture_count = sum(1 for s1 in all_s1_ids if gt_map.get(s1, set()).issubset(union_cand_map.get(s1, set())))
    full_capture_pct = (full_capture_count / len(all_s1_ids)) * 100.0

    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK RESULTS (10,000 S1 Entities)", flush=True)
    print("=" * 70, flush=True)
    print(f"Exact Recall:                     {exact_rec:.2f}% ({exact_cap:,} / {total_true_links:,} links | {exact_pairs:,} pairs)", flush=True)
    print(f"Token Recall:                     {token_rec:.2f}% ({token_cap:,} / {total_true_links:,} links | {token_pairs:,} pairs)", flush=True)
    print(f"Address Recall:                   {addr_rec:.2f}% ({addr_cap:,} / {total_true_links:,} links | {addr_pairs:,} pairs)", flush=True)
    print(f"N-gram Recall:                    {ngram_rec:.2f}% ({ngram_cap:,} / {total_true_links:,} links | {ngram_pairs:,} pairs)", flush=True)
    print("-" * 70, flush=True)
    print(f"Combined Recall (4-Route UNION):  {union_rec:.2f}% ({union_cap:,} / {total_true_links:,} links)", flush=True)
    print(f"Total Candidate Pairs:            {union_pairs:,}", flush=True)
    print(f"Average Candidates per S1:        {np.mean(counts_arr):.2f}", flush=True)
    print(f"P95 Candidates:                   {np.percentile(counts_arr, 95):.1f}", flush=True)
    print(f"P99 Candidates:                   {np.percentile(counts_arr, 99):.1f}", flush=True)
    print(f"Maximum Candidates:               {np.max(counts_arr)}", flush=True)
    print(f"All-True-Matches-Captured Pct:    {full_capture_pct:.2f}% ({full_capture_count:,} / {len(all_s1_ids):,} entities)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_benchmark()
