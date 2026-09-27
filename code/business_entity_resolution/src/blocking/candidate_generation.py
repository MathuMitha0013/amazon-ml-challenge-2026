"""
Multi-route blocking union and high-recall candidate pool generator.

Responsibilities:
- Combine candidate pairs across all blocking routes (Exact, Token, Address, N-Gram).
- Deduplicate candidate pairs and track route provenance (which routes discovered each pair).
- Calculate number_of_blocks_hit for downstream ML matchers.
- Support configurable subsets of routes.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.blocking.exact_blocking import generate_exact_blocks
from src.blocking.token_blocking import generate_token_blocks
from src.blocking.address_blocking import generate_address_blocks
from src.blocking.ngram_blocking import generate_ngram_blocks


def combine_candidate_routes(
    route_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merges candidate pair DataFrames across multiple blocking routes via UNION.
    Tracks block_type provenance and number_of_blocks_hit.

    Args:
        route_dfs: Dictionary mapping route name (e.g. 'exact', 'token', 'address', 'ngram')
                   to DataFrame of ['s1_id', 'matched_id'].

    Returns:
        Deduplicated candidate DataFrame with columns:
        ['s1_id', 'matched_id', 'block_type', 'number_of_blocks_hit']
    """
    tagged_records = []
    for route_name, df in route_dfs.items():
        if df is not None and not df.empty:
            sub = df[["s1_id", "matched_id"]].drop_duplicates().copy()
            sub["route"] = route_name
            tagged_records.append(sub)

    if not tagged_records:
        return pd.DataFrame(columns=["s1_id", "matched_id", "block_type", "number_of_blocks_hit"])

    all_tagged = pd.concat(tagged_records, ignore_index=True)

    # Group by (s1_id, matched_id) to aggregate block types and hits
    grouped = all_tagged.groupby(["s1_id", "matched_id"]).agg(
        block_type=("route", lambda x: "+".join(sorted(set(x)))),
        number_of_blocks_hit=("route", "nunique"),
    ).reset_index()

    return grouped


def generate_candidate_pairs_from_frames(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    routes: Optional[list[str]] = None,
    name_col: str = "name_normalized",
    addr_col: str = "address_normalized",
) -> pd.DataFrame:
    """
    Executes configured blocking routes and merges candidate pools via UNION.

    Args:
        s1_df: Source1 records with s1_id, country, name_normalized, address_normalized.
        target_df: Target records with matched_id, country, name_normalized, address_normalized.
        routes: List of active routes ('exact', 'token', 'address', 'ngram').
        name_col: Column name for normalized business name.
        addr_col: Column name for normalized address.

    Returns:
        Unified candidate DataFrame.
    """
    if routes is None:
        routes = ["exact", "token", "address", "ngram"]

    route_dfs = {}

    if "exact" in routes and name_col in s1_df.columns and name_col in target_df.columns:
        route_dfs["exact"] = generate_exact_blocks(s1_df, target_df, key_column=name_col)

    if "token" in routes and name_col in s1_df.columns and name_col in target_df.columns:
        route_dfs["token"] = generate_token_blocks(s1_df, target_df, max_token_frequency=100, min_token_length=3, key_column=name_col)

    if "address" in routes and addr_col in s1_df.columns and addr_col in target_df.columns:
        route_dfs["address"] = generate_address_blocks(s1_df, target_df, max_token_frequency=75, min_token_length=3, key_column=addr_col)

    if "ngram" in routes and name_col in s1_df.columns and name_col in target_df.columns:
        route_dfs["ngram"] = generate_ngram_blocks(s1_df, target_df, ngram_size=3, max_ngram_frequency=50, key_column=name_col)

    return combine_candidate_routes(route_dfs)


def generate_candidate_pairs(
    source1_path: Union[str, Path, pd.DataFrame],
    source2_path: Union[str, Path, pd.DataFrame],
    source3_path: Optional[Union[str, Path, pd.DataFrame]] = None,
    routes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Executes all configured blocking routes and merges candidate pools via UNION.

    Args:
        source1_path: Path or DataFrame of Source1 dataset.
        source2_path: Path or DataFrame of Source2 dataset.
        source3_path: Optional path or DataFrame of Source3 dataset.
        routes: List of active routes to run (default: ['exact', 'token', 'address', 'ngram']).

    Returns:
        Deduplicated candidate pair table with route hit counts.
    """
    if isinstance(source1_path, (str, Path)):
        s1_df = pd.read_csv(source1_path, sep="\t")
    else:
        s1_df = source1_path.copy()

    if isinstance(source2_path, (str, Path)):
        s2_df = pd.read_csv(source2_path, sep="\t")
    else:
        s2_df = source2_path.copy()

    if source3_path is not None:
        if isinstance(source3_path, (str, Path)):
            s3_df = pd.read_csv(source3_path, sep="\t")
        else:
            s3_df = source3_path.copy()
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    else:
        target_df = s2_df

    return generate_candidate_pairs_from_frames(s1_df, target_df, routes=routes)
