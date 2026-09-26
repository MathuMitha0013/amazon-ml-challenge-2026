"""
Multi-route candidate generation.

Current implemented routes:
    - exact
    - token

The candidate generator combines candidates from Source 2 and Source 3,
deduplicates pairs, and records which blocking routes discovered each pair.
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .exact_blocking import generate_exact_blocks
from .token_blocking import generate_token_blocks


def _load_tsv(path: str | Path) -> pd.DataFrame:
    """Load a TSV file into a DataFrame."""
    return pd.read_csv(path, sep="\t", dtype=str)


def _prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Source 2/3 schema to the common target schema."""
    return df.rename(
        columns={"entity_id": "matched_id"}
    )


def _run_route(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    route: str,
) -> pd.DataFrame:
    """Run one blocking route."""
    route = route.lower()

    if route == "exact":
        result = generate_exact_blocks(source1, target)

    elif route == "token":
        result = generate_token_blocks(
            source1,
            target,
            max_token_frequency=100,
            min_token_length=3,
        )

    else:
        raise ValueError(
            f"Unsupported route '{route}'. "
            f"Currently supported routes: exact, token"
        )

    result = result[
        ["s1_id", "matched_id"]
    ].drop_duplicates()

    result["block_type"] = route

    return result


def generate_candidate_pairs(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    routes: Optional[list[str]] = None,
) -> Any:
    """
    Generate candidate pairs across Source 2 and Source 3.

    Args:
        source1_path: Path to Source 1 TSV.
        source2_path: Path to Source 2 TSV.
        source3_path: Path to Source 3 TSV.
        routes: Blocking routes to run.
                Default: ["exact", "token"].

    Returns:
        DataFrame with:
            s1_id
            matched_id
            block_type
            number_of_blocks_hit

    Notes:
        Address and n-gram routes are intentionally not enabled yet.
    """

    if routes is None:
        routes = ["exact", "token"]

    source1 = _load_tsv(source1_path)
    source2 = _prepare_target(_load_tsv(source2_path))
    source3 = _prepare_target(_load_tsv(source3_path))

    # The normalization pipeline should provide this column.
    required_s1 = {"entity_id", "country", "name_normalized"}

    if not required_s1.issubset(source1.columns):
        missing = required_s1 - set(source1.columns)
        raise ValueError(
            f"Source 1 is missing required columns: {sorted(missing)}"
        )

    source1 = source1.rename(columns={"entity_id": "s1_id"})

    required_target = {
        "matched_id",
        "country",
        "name_normalized",
    }

    for name, df in [("Source 2", source2), ("Source 3", source3)]:
        if not required_target.issubset(df.columns):
            missing = required_target - set(df.columns)
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    all_results = []

    # Source 2
    for route in routes:
        result = _run_route(source1, source2, route)
        result["source"] = "S2"
        all_results.append(result)

    # Source 3
    for route in routes:
        result = _run_route(source1, source3, route)
        result["source"] = "S3"
        all_results.append(result)

    if not all_results:
        return pd.DataFrame(
            columns=[
                "s1_id",
                "matched_id",
                "source",
                "block_type",
                "number_of_blocks_hit",
            ]
        )

    candidates = pd.concat(
        all_results,
        ignore_index=True,
    )

    # A pair may be discovered by multiple blocking routes.
    # Combine route names and count the number of routes.
    final = (
        candidates
        .groupby(
            ["s1_id", "matched_id", "source"],
            as_index=False,
        )
        .agg(
            block_type=(
                "block_type",
                lambda x: ",".join(sorted(set(x))),
            ),
            number_of_blocks_hit=(
                "block_type",
                "nunique",
            ),
        )
    )

    return final
