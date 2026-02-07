"""
Self-contained versioning utilities for replication experiments.
Adapted from parent versioning library to be standalone.
"""
import time
import json
import pandas as pd
import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from delphi_epidata import Epidata

# Local imports
from replication.config import DATA_DIR, DELPHI_API_KEY

load_dotenv()
Epidata.auth = ("epidata", DELPHI_API_KEY)  # type:ignore

DELTA_DIR = DATA_DIR / "deltas"
API_ENDPOINT = {"wili": Epidata.fluview}

# ============================================================================
# UTILITIES
# ============================================================================

def step_epiweeks(epiweeks: list, current_yw: int, k: int) -> int | None:
    """Step k weeks forward or backward in the epiweek list."""
    try:
        current_idx = epiweeks.index(current_yw)
        target_idx = current_idx + k

        if target_idx >= len(epiweeks) or target_idx < 0:
            return None

        return epiweeks[target_idx]
    except ValueError:
        logger.error(f"Epiweek {current_yw} not found in series.")
        return None


def earliest_active_epiweek(
    epiweeks: list[int], lookup: dict[int, int], issue: int, start_yw: int
):
    """Find earliest epiweek where data was revised after the given issue."""
    i0 = epiweeks.index(start_yw)
    for t in epiweeks[i0:]:
        if lookup[t] > issue:
            logger.info(
                f"Issue {issue} is active as of epiweek={t} with max issue {lookup[t]} > {issue}."
            )
            return t
    return None


def check_missing_epiweek(
    epiweeks: list[int], returned_epiweeks: list[int], start_yw: int, end_yw: int
):
    """Check for missing epiweeks in API response."""
    start_idx, end_idx = epiweeks.index(start_yw), epiweeks.index(end_yw)
    if set(returned_epiweeks) != set(epiweeks[start_idx : end_idx + 1]):
        missing_weeks = list(
            set(epiweeks[start_idx : end_idx + 1]) - set(returned_epiweeks)
        )
        logger.info(
            f"Queried {start_yw}-{end_yw}, fixed to issue={end_yw} returned {min(returned_epiweeks)}-{max(returned_epiweeks)}."
        )
        if missing_weeks != []:
            logger.warning(
                f"Missing epiweeks detected from issue={end_yw} query: {missing_weeks}."
            )
        return missing_weeks
    return []


# ============================================================================
# DATA CONSTRUCTION
# ============================================================================

def pull_finalized_series(
    start_yw: int, end_yw: int, series: str = "wili", region: str = "nat"
) -> pd.DataFrame:
    """Pull finalized series for a specific region."""
    api_fn = API_ENDPOINT[series]
    res = api_fn([region], epiweeks=Epidata.range(start_yw, end_yw))["epidata"]  # type: ignore
    return pd.DataFrame(res)


def pull_finalized_info(df) -> tuple[dict[int, int], list[int]]:
    """Extract max issue lookup and epiweek list from finalized data."""
    max_issue_lookup = {
        item["epiweek"]: item["issue"] for item in df.to_dict(orient="records")
    }
    epiweeks = df["epiweek"].sort_values().tolist()
    logger.info(
        f"Finalized series. Max epiweek: {max(epiweeks)}, min epiweek: {min(epiweeks)}"
    )
    return max_issue_lookup, epiweeks


def build_base(df: pd.DataFrame, initial_issue: int) -> pd.DataFrame:
    """Build base series up to initial issue date."""
    is_finalized = df["issue"] <= initial_issue
    base = df[is_finalized]
    logger.info(
        f"Base series. Max epiweek: {base['epiweek'].max()}, min epiweek: {base['epiweek'].min()}"
    )
    return base


def calculate_delta(
    issue: int,
    start_yw: int,
    epiweeks: list[int],
    series: str = "wili",
    region: str = "nat",
):
    """Calculate delta (revisions) for a specific issue."""
    api_fn = API_ENDPOINT[series]
    res = api_fn(
        [region], epiweeks=Epidata.range(start_yw, issue), issues=issue
    )["epidata"]  # type: ignore

    # Check if epidata is None or empty list (as observed in debug for issue 202350)
    if not res:
        logger.warning(f"Issue {issue} returned no data for region={region}. Skipping.")
        return None

    df = pd.DataFrame(res)
    check_missing_epiweek(
        epiweeks=epiweeks,
        returned_epiweeks=df["epiweek"].tolist(),
        start_yw=start_yw,
        end_yw=issue,
    )

    logger.info(f"Delta for issue={issue} has date range [{start_yw}, {issue}].")
    return pd.DataFrame(res)


def build_asof(
    target_issue: int,
    delta_metadata: dict,
    max_issue_lookup: dict,
    base_series: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct time series as it appeared at a specific issue date."""

    def step_back_one_epiweek(current_yw: int) -> int:
        year, week = divmod(current_yw, 100)
        return (year - 1) * 100 + 52 if week == 1 else current_yw - 1

    result = []
    current_issue = target_issue
    current_start_yw = None

    # Get base series coverage
    base_max_epiweek = base_series["epiweek"].max()

    while (
        current_start_yw is None or current_start_yw > base_max_epiweek
    ) and str(current_issue) in delta_metadata:

        file_path = DELTA_DIR / f"asof={current_issue}.parquet"

        # Extra safety: stop if the file is missing
        if not file_path.exists():
            logger.warning(
                f"Expected delta file missing: {file_path}. Stopping traversal."
            )
            break

        delta = pd.read_parquet(file_path)

        if current_start_yw is not None:
            delta = delta[delta["epiweek"] < current_start_yw]

        result.append(delta)

        # Get metadata for next jump
        metadata = delta_metadata[str(current_issue)]
        current_start_yw = metadata["start_epiweek"]

        prev_yw = step_back_one_epiweek(current_start_yw)

        # JUMP: If the lookup doesn't exist, we must break
        current_issue = max_issue_lookup.get(str(prev_yw))
        if current_issue is None:
            break

    result.append(base_series)
    return pd.concat(result).sort_values("epiweek").reset_index(drop=True)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def generate_versioning_data(
    region: str,
    start_epiweek: int = 201650,
    end_epiweek: int = 202352,
    init_train: int = 128,
    api_sleep: float = 1.0,
):
    """
    Generate versioning data (base, deltas, metadata) for a specific region.

    This overwrites existing versioning files in DATA_DIR.

    Args:
        region: Region code (e.g., 'nat', 'al', 'ca')
        start_epiweek: Start of observation period
        end_epiweek: End of observation period
        init_train: Initial training window size
        api_sleep: Sleep time between API calls
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating versioning data for region: {region}")

    # Pull finalized series
    finalized_df = pull_finalized_series(
        start_yw=start_epiweek, end_yw=end_epiweek, region=region
    )
    finalized_df.to_parquet(DATA_DIR / "finalized.parquet")

    # Extract metadata
    MAX_LOOKUP_TABLE, EPIWEEKS = pull_finalized_info(finalized_df)
    with open(DATA_DIR / "max_issue_lookup.json", "w") as f:
        json.dump(MAX_LOOKUP_TABLE, f, indent=4)

    # Build base series
    init_issue = step_epiweeks(EPIWEEKS, start_epiweek, init_train)
    base_series = build_base(finalized_df, init_issue)  # type:ignore
    base_series.to_parquet(DATA_DIR / "base.parquet")

    # Build deltas
    start_yw = step_epiweeks(EPIWEEKS, int(base_series["epiweek"].max()), 1)
    logger.info(f"Initial start_yw {start_yw}")

    delta_metadata = {}
    for w in EPIWEEKS[EPIWEEKS.index(init_issue) :]:  # type:ignore
        delta = calculate_delta(
            issue=w, start_yw=start_yw, epiweeks=EPIWEEKS, region=region  # type:ignore
        )

        # Skip this issue if API returned no data
        if delta is None:
            time.sleep(api_sleep)
            continue

        file_path = DELTA_DIR / f"asof={w}.parquet"
        delta.to_parquet(file_path)

        delta_metadata[int(w)] = {
            "start_epiweek": int(start_yw),  # type:ignore
            "end_epiweek": int(w),
            "filename": file_path.name,
            "rows": len(delta),
        }

        time.sleep(api_sleep)

        start_yw = earliest_active_epiweek(
            EPIWEEKS, MAX_LOOKUP_TABLE, w, start_yw  # type:ignore
        )
        if start_yw is None:
            logger.warning(f"issue={w} leads to start_yw={start_yw}")
            break

    with open(DATA_DIR / "delta_metadata.json", "w") as f:
        json.dump(delta_metadata, f, indent=4)

    logger.info(f"Versioning data generation complete for region: {region}")
