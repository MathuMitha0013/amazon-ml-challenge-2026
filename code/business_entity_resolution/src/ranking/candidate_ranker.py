"""
Candidate pool ranking, heuristic scoring, and adaptive budget truncation.

Provides a scalable, vectorized candidate ranker that operates on candidate pairs
generated from blocking routes, computes lightweight similarity signals, and truncates
candidate pools to a configurable top-K budget per Source1 entity.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence
import polars as pl
import numpy as np
from rapidfuzz import fuzz


def _extract_digits_set(text: Optional[str]) -> set[str]:
    """Extract all numeric digit sequences from string."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def _token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(intersection / union) if union > 0 else 0.0


def _tokenize_simple(text: Optional[str]) -> set[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return set()
    return set(re.findall(r"\w+", text.lower()))


from ..features.pair_features import standardize_candidate_schema, hydrate_candidate_pairs


class CandidateRanker:
    """
    Lightweight, high-throughput candidate ranker for S1-(S2/S3) pairs.

    Computes fast string, numeric, and blocking metadata signals to rank candidate
    pairs and retain the top-K highest-probability candidates per Source1 entity.
    """

    def __init__(
        self,
        top_k: int = 30,
        min_score: float = 0.0,
        s1_id_col: str = "source1_entity_id",
        candidate_id_col: str = "candidate_entity_id",
        s1_name_col: str = "s1_business_name",
        candidate_name_col: str = "candidate_business_name",
        s1_address_col: str = "s1_business_address",
        candidate_address_col: str = "candidate_business_address",
        s1_country_col: str = "s1_country",
        candidate_country_col: str = "candidate_country",
        block_type_col: Optional[str] = "block_type",
        num_blocks_hit_col: Optional[str] = "number_of_blocks_hit",
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the CandidateRanker.
        """
        self.top_k = top_k
        self.min_score = min_score

        self.s1_id_col = s1_id_col
        self.candidate_id_col = candidate_id_col
        self.s1_name_col = s1_name_col
        self.candidate_name_col = candidate_name_col
        self.s1_address_col = s1_address_col
        self.candidate_address_col = candidate_address_col
        self.s1_country_col = s1_country_col
        self.candidate_country_col = candidate_country_col
        self.block_type_col = block_type_col
        self.num_blocks_hit_col = num_blocks_hit_col

        self.weights = weights or {
            "name_ratio": 0.40,
            "name_token_jaccard": 0.20,
            "address_ratio": 0.15,
            "address_token_jaccard": 0.10,
            "numeric_overlap": 0.05,
            "country_match": 0.05,
            "block_hit_bonus": 0.05,
        }

    def compute_ranking_signals(
        self,
        df: Any,
        source1_table: Optional[Any] = None,
        target_table: Optional[Any] = None,
    ) -> pl.DataFrame:
        """
        Calculates fast ranking signals for each candidate pair in the DataFrame.
        Automatically standardizes schemas (s1_id -> source1_entity_id, matched_id -> candidate_entity_id)
        and hydrates text attributes if source tables are provided.

        Args:
            df: Candidate pairs DataFrame.
            source1_table: Optional Source1 table to join attributes.
            target_table: Optional target table to join attributes.

        Returns:
            Polars DataFrame augmented with ranking score components and `rank_score`.
        """
        df = standardize_candidate_schema(df)

        if source1_table is not None and self.s1_name_col not in df.columns:
            df = hydrate_candidate_pairs(df, source1_table=source1_table, target_table=target_table)

        if df.height == 0:
            return df.with_columns(pl.lit(0.0).alias("rank_score"))

        # Extract text columns safely
        s1_names = df[self.s1_name_col].fill_null("").to_list() if self.s1_name_col in df.columns else [""] * df.height
        c_names = df[self.candidate_name_col].fill_null("").to_list() if self.candidate_name_col in df.columns else [""] * df.height

        s1_addrs = df[self.s1_address_col].fill_null("").to_list() if self.s1_address_col in df.columns else [""] * df.height
        c_addrs = df[self.candidate_address_col].fill_null("").to_list() if self.candidate_address_col in df.columns else [""] * df.height

        s1_countries = df[self.s1_country_col].fill_null("").to_list() if self.s1_country_col in df.columns else [""] * df.height
        c_countries = df[self.candidate_country_col].fill_null("").to_list() if self.candidate_country_col in df.columns else [""] * df.height

        has_hit_col = self.num_blocks_hit_col and self.num_blocks_hit_col in df.columns
        hit_counts = df[self.num_blocks_hit_col].fill_null(1).to_list() if has_hit_col else [1] * df.height

        n = df.height
        name_ratios = np.zeros(n, dtype=np.float32)
        name_jaccards = np.zeros(n, dtype=np.float32)
        addr_ratios = np.zeros(n, dtype=np.float32)
        addr_jaccards = np.zeros(n, dtype=np.float32)
        numeric_overlaps = np.zeros(n, dtype=np.float32)
        country_matches = np.zeros(n, dtype=np.float32)
        hit_bonuses = np.zeros(n, dtype=np.float32)

        for i in range(n):
            sn, cn = s1_names[i], c_names[i]
            sa, ca = s1_addrs[i], c_addrs[i]
            sc, cc = s1_countries[i], c_countries[i]

            # RapidFuzz ratio (normalized to 0-1)
            if sn and cn:
                name_ratios[i] = fuzz.ratio(sn, cn) / 100.0
                sn_toks = _tokenize_simple(sn)
                cn_toks = _tokenize_simple(cn)
                name_jaccards[i] = _token_jaccard(sn_toks, cn_toks)

            if sa and ca:
                addr_ratios[i] = fuzz.ratio(sa, ca) / 100.0
                sa_toks = _tokenize_simple(sa)
                ca_toks = _tokenize_simple(ca)
                addr_jaccards[i] = _token_jaccard(sa_toks, ca_toks)

                sa_nums = _extract_digits_set(sa)
                ca_nums = _extract_digits_set(ca)
                if sa_nums and ca_nums:
                    numeric_overlaps[i] = float(len(sa_nums & ca_nums) / len(sa_nums | ca_nums))

            # Country match (1.0 for match, 0.5 for missing/unspecified, 0.0 for mismatch)
            if not sc or not cc:
                country_matches[i] = 0.5
            elif sc.strip().upper() == cc.strip().upper():
                country_matches[i] = 1.0
            else:
                country_matches[i] = 0.0

            # Hit count bonus (capped at 5 routes)
            hit_bonuses[i] = min(hit_counts[i] / 5.0, 1.0)

        # Composite score
        w = self.weights
        composite_score = (
            w.get("name_ratio", 0.40) * name_ratios
            + w.get("name_token_jaccard", 0.20) * name_jaccards
            + w.get("address_ratio", 0.15) * addr_ratios
            + w.get("address_token_jaccard", 0.10) * addr_jaccards
            + w.get("numeric_overlap", 0.05) * numeric_overlaps
            + w.get("country_match", 0.05) * country_matches
            + w.get("block_hit_bonus", 0.05) * hit_bonuses
        )

        return df.with_columns(
            [
                pl.Series("rank_score", composite_score),
                pl.Series("name_ratio", name_ratios),
                pl.Series("addr_ratio", addr_ratios),
                pl.Series("country_match", country_matches),
            ]
        )

    def rank_and_truncate(
        self,
        df: Any,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        source1_table: Optional[Any] = None,
        target_table: Optional[Any] = None,
    ) -> pl.DataFrame:
        """
        Ranks candidate pairs and truncates to top-K per Source1 entity.

        Args:
            df: Input candidate pairs DataFrame.
            top_k: Optional override for top_k.
            min_score: Optional override for minimum score threshold.
            source1_table: Optional Source1 table for attribute hydration.
            target_table: Optional target table for attribute hydration.

        Returns:
            Truncated Polars DataFrame with ranked candidate pairs.
        """
        df = standardize_candidate_schema(df)
        k = top_k if top_k is not None else self.top_k
        score_cutoff = min_score if min_score is not None else self.min_score

        if "rank_score" not in df.columns:
            df = self.compute_ranking_signals(df, source1_table=source1_table, target_table=target_table)

        if score_cutoff > 0.0:
            df = df.filter(pl.col("rank_score") >= score_cutoff)

        if df.height == 0:
            return df

        # Sort descending by rank_score per S1 ID and retain top-K
        ranked_df = (
            df.sort([self.s1_id_col, "rank_score"], descending=[False, True])
            .group_by(self.s1_id_col, maintain_order=True)
            .head(k)
        )

        return ranked_df

    def format_candidate_pairs_tsv(
        self,
        ranked_df: pl.DataFrame,
        all_s1_ids: Optional[Sequence[str]] = None,
    ) -> pl.DataFrame:
        """
        Formats ranked candidates into the official candidate_pairs.tsv structure:
        [source1_entity_id, candidate_entity_ids]

        Args:
            ranked_df: DataFrame with ranked candidates.
            all_s1_ids: Complete set of required test S1 IDs (ensures every S1 entity appears).

        Returns:
            Polars DataFrame with exact required candidate_pairs columns.
        """
        # Group candidates by S1 ID
        grouped = (
            ranked_df.group_by(self.s1_id_col, maintain_order=True)
            .agg(pl.col(self.candidate_id_col).unique(maintain_order=True).alias("cand_list"))
            .select(
                [
                    pl.col(self.s1_id_col).alias("source1_entity_id"),
                    pl.col("cand_list").list.join(",").alias("candidate_entity_ids"),
                ]
            )
        )

        if all_s1_ids is not None:
            full_s1_df = pl.DataFrame({"source1_entity_id": list(all_s1_ids)})
            grouped = full_s1_df.join(grouped, on="source1_entity_id", how="left").with_columns(
                pl.col("candidate_entity_ids").fill_null("")
            )

        return grouped


def rank_candidates(
    candidate_pairs: pl.DataFrame,
    top_k: int = 30,
    min_score_threshold: float = 0.0,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Convenience functional wrapper around CandidateRanker.

    Args:
        candidate_pairs: Polars DataFrame of candidate pairs.
        top_k: Maximum candidate pairs per S1.
        min_score_threshold: Minimum rank score.
        kwargs: Additional arguments passed to CandidateRanker.

    Returns:
        Ranked Polars DataFrame.
    """
    ranker = CandidateRanker(top_k=top_k, min_score=min_score_threshold, **kwargs)
    return ranker.rank_and_truncate(candidate_pairs)


def allocate_candidate_budget(
    ranked_candidates: pl.DataFrame,
    strategy: str = "fixed_k",
    top_k: int = 30,
    min_score: float = 0.05,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Applies budgeting strategy to candidate pool.

    Args:
        ranked_candidates: Scored candidate pairs.
        strategy: 'fixed_k', 'threshold', or 'adaptive'.
        top_k: Top-K budget.
        min_score: Minimum score threshold.

    Returns:
        Filtered candidate pairs.
    """
    ranker = CandidateRanker(top_k=top_k, min_score=min_score, **kwargs)
    return ranker.rank_and_truncate(ranked_candidates, top_k=top_k, min_score=min_score)
