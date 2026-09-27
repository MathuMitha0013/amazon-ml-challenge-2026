import os
import json
import duckdb
import numpy as np
import polars as pl

def run_ground_truth_analysis(train_s1_path, train_s2_path, train_s3_path, gt_path, output_json_path):
    print("--- Running Stage 1 Ground Truth & Data Profiling Analysis ---")

    con = duckdb.connect()

    # Load Ground Truth
    gt_df = pl.read_csv(gt_path, separator="\t")

    # Explode matched_entity_ids to get pairs
    gt_exploded = gt_df.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids")

    # Filter non-empty matches
    gt_valid = gt_exploded.filter(
        (pl.col("matched_entity_ids").is_not_null()) &
        (pl.col("matched_entity_ids") != "")
    )

    # Compute match counts per S1 entity
    match_counts_df = gt_df.with_columns(
        pl.when((pl.col("matched_entity_ids").is_null()) | (pl.col("matched_entity_ids") == ""))
        .then(0)
        .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())
        .alias("match_count")
    )

    counts = match_counts_df["match_count"].to_numpy()

    # Statistics
    total_s1 = len(match_counts_df)
    singleton_count = int(np.sum(counts == 0))
    singleton_pct = float((singleton_count / total_s1) * 100)

    avg_matches = float(np.mean(counts))
    p50 = float(np.percentile(counts, 50))
    p90 = float(np.percentile(counts, 90))
    p95 = float(np.percentile(counts, 95))
    p99 = float(np.percentile(counts, 99))
    max_matches = int(np.max(counts))

    # Source 2 vs Source 3 distribution
    s2_matches = gt_valid.filter(pl.col("matched_entity_ids").str.starts_with("S2-"))
    s3_matches = gt_valid.filter(pl.col("matched_entity_ids").str.starts_with("S3-"))

    # Country Analysis using DuckDB
    con.execute(f"""
        CREATE VIEW s1 AS SELECT entity_id, country FROM read_csv_auto('{train_s1_path}', delim='\t');
        CREATE VIEW s2 AS SELECT entity_id, country FROM read_csv_auto('{train_s2_path}', delim='\t');
        CREATE VIEW s3 AS SELECT entity_id, country FROM read_csv_auto('{train_s3_path}', delim='\t');
        CREATE VIEW target AS SELECT * FROM s2 UNION ALL SELECT * FROM s3;
    """
    )

    gt_valid_pd = gt_valid.to_pandas()
    con.register("gt_pairs", gt_valid_pd)

    cross_country_query = """
        SELECT
            COUNT(*) as total_valid_matches,
            SUM(CASE WHEN s1.country != target.country THEN 1 ELSE 0 END) as cross_country_matches
        FROM gt_pairs g
        JOIN s1 ON g.source1_entity_id = s1.entity_id
        JOIN target ON g.matched_entity_ids = target.entity_id
    """
    res = con.execute(cross_country_query).fetchone()
    total_valid_pairs = res[0] if res else 0
    cross_country_cnt = res[1] if res else 0
    cross_country_pct = float((cross_country_cnt / total_valid_pairs) * 100) if total_valid_pairs > 0 else 0.0

    country_dist = con.execute("""
        SELECT country, COUNT(*) as count
        FROM (SELECT country FROM s1 UNION ALL SELECT country FROM s2 UNION ALL SELECT country FROM s3)
        GROUP BY country
    """
    ).fetchall()
    country_distribution = {c[0]: c[1] for c in country_dist}

    # Strange Ground Truth Cases
    dup_within_list = 0
    for row in gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and isinstance(m_str, str):
            ids = m_str.split(",")
            if len(ids) != len(set(ids)):
                dup_within_list += 1

    target_multi_mapped = gt_valid.group_by("matched_entity_ids").agg(
        pl.count("source1_entity_id").alias("s1_count")
    ).filter(pl.col("s1_count") > 1)

    profile_results = {
        "dataset_summary": {
            "total_source1_entities": total_s1,
            "total_valid_match_pairs": total_valid_pairs
        },
        "exact_stage1_statistics": {
            "Singleton %": f"{singleton_pct:.2f}%",
            "Average matches": round(avg_matches, 4),
            "P50": p50,
            "P90": p90,
            "P95": p95,
            "P99": p99,
            "Maximum matches": max_matches,
            "Country distribution": country_distribution,
            "Cross-country matches": f"{cross_country_cnt} ({cross_country_pct:.2f}%)"
        },
        "s2_s3_breakdown": {
            "S2_matches_count": len(s2_matches),
            "S3_matches_count": len(s3_matches)
        },
        "strange_ground_truth_cases": {
            "rows_with_duplicate_ids_in_same_list": dup_within_list,
            "target_ids_mapped_to_multiple_s1_entities": len(target_multi_mapped)
        }
    }

    with open(output_json_path, "w") as f:
        json.dump(profile_results, f, indent=4)

    print(f"Profiling complete! File saved at: {output_json_path}")
    return profile_results
