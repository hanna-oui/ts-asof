"""Source-agnostic delta construction and as-of reconstruction engine.

Build phase:  fetch a finalized snapshot, split by entity (if panel data),
construct a base series and iteratively store revision deltas.

Reconstruct phase:  given a target version, traverse stored deltas backward
to reassemble the series as it appeared at that version.
"""

from __future__ import annotations

import json
import logging
from tqdm import tqdm  # Add this to your imports
import time
from pathlib import Path

import pandas as pd

from asof.sources.base import DataSource
from asof.utils import check_missing_indices, earliest_active_index, step_index

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────


def extract_finalized_info(
    df: pd.DataFrame,
    time_col: str,
    version_col: str,
) -> tuple[dict[str, int], list[int]]:
    """Extract a *max-version lookup* and the sorted time index from a
    finalized snapshot. """

    """
    ASSUMPTION: in your finalized df, you have a mapping between
    when the last issue date is and the corresponding date
    """
    # Use string keys so the dict matches JSON serialization format
    max_version_lookup = {
        str(row[time_col]): int(row[version_col])
        for row in df.to_dict(orient="records")
    }

    sorted_time_indices = df[time_col].sort_values().tolist()
    logger.info(
        "Snapshot info: %d indices, range [%s, %s]",
        len(sorted_time_indices),
        min(sorted_time_indices),
        max(sorted_time_indices),
    )
    return max_version_lookup, sorted_time_indices


def build_base(
    finalized_df: pd.DataFrame,
    initial_version: int,
    version_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Filter *df* to epiweeks fully finalized before *initial_version*.

    An epiweek is fully finalized when its final issue date (max version)
    is strictly less than *initial_version*, meaning it will never be
    revised again in subsequent delta versions.
    """
    final_issue = finalized_df.groupby(time_col)[version_col].max()
    finalized_times = final_issue[final_issue < initial_version].index
    base = finalized_df[finalized_df[time_col].isin(finalized_times)].copy()
    logger.info(
        "Base: %d rows, %s range [%s, %s]",
        len(base),
        time_col,
        base[time_col].min(),
        base[time_col].max(),
    )
    return base


# ── single-entity build ───────────────────────────────────────────────────


def _build_single_entity(
    source: DataSource,
    finalized_df: pd.DataFrame,
    target_dir: Path,
    start: int,
    init_window: int, 
    api_sleep: float,
) -> None:
    
    # Use the source properties internally
    time_col = source.time_col
    version_col = source.version_col
    
    delta_dir = target_dir / "deltas"
    target_dir.mkdir(parents=True, exist_ok=True)
    delta_dir.mkdir(parents=True, exist_ok=True)

    finalized_df.to_parquet(target_dir / "finalized.parquet")

    max_version_dict, time_indices = extract_finalized_info(finalized_df, time_col, version_col)
    with open(target_dir / "max_version_lookup.json", "w") as f:
        json.dump(max_version_dict, f, indent=4)

    # Use init_window - 1 to get the N-th element (e.g., window of 20 → index 19)
    init_version = step_index(time_indices, start, init_window - 1)
    if init_version is None:
        raise ValueError(
            f"init_window={init_window} exceeds available indices "
            f"({len(time_indices)}). Use a value <= {len(time_indices)}."
        )

    base_series = build_base(finalized_df, init_version, version_col, time_col)
    base_series.to_parquet(target_dir / "base.parquet")

    # Correct the fallback start (The "Super-Delta")
    if base_series.empty:
        start_yw = start  # Start at the very beginning requested
    else:
        # Pick up immediately after the frozen base
        start_yw = step_index(time_indices, int(base_series[time_col].max()), 1)
        if start_yw is None:
            raise ValueError("No index value after base series maximum.")
    logger.info("Initial delta start: %s", start_yw)

    # load previously-saved metadata so resumption doesn't discard it
    meta_path = target_dir / "delta_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            delta_metadata: dict[int, dict] = {
                int(k): v for k, v in json.load(f).items()
            }
    else:
        delta_metadata: dict[int, dict] = {}

    versions_to_process = time_indices[time_indices.index(init_version):]

    for w in tqdm(versions_to_process, desc="Building cache", unit="version"):
        file_path = delta_dir / f"asof={w}.parquet"

        # resumption: skip already-cached deltas
        if file_path.exists():
            logger.info("version=%d already cached, skipping.", w)
            start_yw = earliest_active_index(time_indices, max_version_dict,
                                             w, start_yw)
            if start_yw is None:
                logger.warning("version=%d → start=None, stopping.", w)
                break
            continue

        logger.info("Fetching version=%d …", w)
        delta = source.fetch_revision(version=w, start=start_yw, end=w)

        if delta is None:
            logger.info("[%d/%d] version=%d returned no data.", w)
            time.sleep(api_sleep)
            continue

        check_missing_indices(
            indices=time_indices,
            returned_indices=delta[time_col].tolist(),
            start=start_yw,
            end=w,
        )

        delta.to_parquet(file_path)

        delta_metadata[int(w)] = {
            f"start_{time_col}": int(start_yw),
            f"end_{time_col}": int(w),
            "filename": file_path.name,
            "rows": len(delta),
        }

        with open(meta_path, "w") as f:
            json.dump(delta_metadata, f, indent=4)

        time.sleep(api_sleep)

        start_yw = earliest_active_index(time_indices, 
                                         max_version_dict, w, start_yw)
        if start_yw is None:
            logger.warning("version=%d, start=None, stopping.", w)
            break

    with open(target_dir / "delta_metadata.json", "w") as f:
        json.dump(delta_metadata, f, indent=4)


# ── multi-entity build ────────────────────────────────────────────────────

def _build_multi_entity(
    source: DataSource,
    finalized_df: pd.DataFrame,
    start: int,
    init_window: int, 
    cache_dir: Path,
    api_sleep: float,
) -> None:
    """Build per-entity caches from a panel snapshot."""
    
    # Extract properties from source once
    time_col = source.time_col
    version_col = source.version_col
    entity_col = source.entity_col 

    
    # sort entities
    entities = sorted(finalized_df[entity_col].unique().tolist())
    entities_root = cache_dir / "entities"

    # ── per-entity setup ──────────────────────────────────────────────
    states: dict[str, dict] = {}
    for entity_val in entities:
        entity_df = finalized_df[finalized_df[entity_col] == entity_val].copy()
        entity_dir = entities_root / str(entity_val)
        delta_dir = entity_dir / "deltas"
        entity_dir.mkdir(parents=True, exist_ok=True)
        delta_dir.mkdir(parents=True, exist_ok=True)

        entity_df.to_parquet(entity_dir / "finalized.parquet")

        max_version_dict, time_indices = extract_finalized_info(entity_df, 
                                                  time_col, version_col)
        with open(entity_dir / "max_version_lookup.json", "w") as f:
            json.dump(max_version_dict, f, indent=4)

        # initial version we need to start with (window of N → N-1 steps)
        init_issue = step_index(time_indices, start, init_window - 1)
        if init_issue is None:
            raise ValueError(
                f"init_window={init_window} exceeds available indices "
                f"({len(time_indices)}) for entity={entity_val}. "
                f"Use a value <= {len(time_indices)}."
            )

        base = build_base(entity_df, init_issue, version_col, time_col)
        base.to_parquet(entity_dir / "base.parquet")

        if base.empty:
            start_yw = start  # Start at the very beginning requested
        else:
            # Pick up immediately after the frozen base
            start_yw = step_index(time_indices, int(base[time_col].max()), 1)
            if start_yw is None:
                raise ValueError(
                    f"No index after base max for entity={entity_val}"
                )

        # load previously-saved metadata so resumption doesn't discard it
        existing_meta_path = entity_dir / "delta_metadata.json"
        if existing_meta_path.exists():
            with open(existing_meta_path) as f:
                existing_meta = {
                    int(k): v for k, v in json.load(f).items()
                }
        else:
            existing_meta = {}

        states[str(entity_val)] = {
            "max_version_lookup": max_version_dict,
            "indices": time_indices,
            "init_issue": init_issue,
            "start_yw": start_yw,
            "delta_metadata": existing_meta,
            "entity_dir": entity_dir,
        }

    # Use first entity's indices for global iteration order
    first = next(iter(states.values()))
    global_indices: list[int] = first["indices"]
    global_init: int = first["init_issue"]

    # ── delta loop: one fetch per version ─────────────────────────────
    versions_to_process = global_indices[global_indices.index(global_init):]

    for w in tqdm(versions_to_process, desc=f"Building cache ({len(entities)} entities)", unit="version"):
        # widest start needed across active entities
        active_starts = [
            s["start_yw"]
            for s in states.values()
            if s["start_yw"] is not None
        ]
        if not active_starts:
            logger.info("All entities exhausted at version=%d.", w)
            break

        min_start = min(active_starts)

        logger.info("Fetching version=%d …", w)
        revision_df = source.fetch_revision(version=w, start=min_start, end=w)

        if revision_df is None:
            logger.info("version=%d returned no data.", w)
            time.sleep(api_sleep)
            continue

        # distribute per entity
        for entity_val, state in states.items():
            if state["start_yw"] is None:
                continue

            entity_rev = revision_df[revision_df[entity_col] == entity_val]
            if entity_rev.empty:
                continue

            file_path = state["entity_dir"] / "deltas" / f"asof={w}.parquet"

            if file_path.exists():
                state["start_yw"] = earliest_active_index(
                    state["indices"],
                    state["max_version_lookup"],
                    w,
                    state["start_yw"],
                )
                continue

            # validate against this entity's expected range
            entity_start = state["start_yw"]
            entity_returned = entity_rev[time_col].tolist()
            check_missing_indices(
                indices=state["indices"],
                returned_indices=entity_returned,
                start=entity_start,
                end=w,
            )

            # filter to this entity's actual range
            entity_rev = entity_rev[entity_rev[time_col] >= entity_start]
            entity_rev.to_parquet(file_path)

            state["delta_metadata"][int(w)] = {
                f"start_{time_col}": int(entity_start),
                f"end_{time_col}": int(w),
                "filename": file_path.name,
                "rows": len(entity_rev),
            }

            with open(state["entity_dir"] / "delta_metadata.json", "w") as f:
                json.dump(state["delta_metadata"], f, indent=4)

            state["start_yw"] = earliest_active_index(
                state["indices"],
                state["max_version_lookup"],
                w,
                state["start_yw"],
            )

        time.sleep(api_sleep)

    # ── save metadata per entity ──────────────────────────────────────
    for state in states.values():
        with open(state["entity_dir"] / "delta_metadata.json", "w") as f:
            json.dump(state["delta_metadata"], f, indent=4)

    logger.info("Multi-entity build complete: %d entities", len(entities))


# ── public entry point ─────────────────────────────────────────────────────


def build_cache(
    source: DataSource,
    start_date: int,
    end_date: int,
    init_window_size: int, 
    cache_dir: Path,
    api_sleep: float = 1.0,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Persist source metadata using the source properties directly
    meta = {
        "time_col": source.time_col, 
        "version_col": source.version_col, 
        "entity_col": source.entity_col
    }
    with open(cache_dir / "source_meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    # Fetch finalized df once
    finalized_df = source.fetch_finalized(start=start_date, end=end_date)
    
    # Decide based on the ACTUAL count of IDs
    multi_entity = source.entity_col is not None and len(source.entity_ids) > 1

    if not multi_entity:
        _build_single_entity(
            source=source,
            finalized_df=finalized_df,
            target_dir=cache_dir,
            start=start_date,
            init_window=init_window_size, 
            api_sleep=api_sleep,
        )
    else:
        _build_multi_entity(
            source=source,
            finalized_df=finalized_df,
            cache_dir=cache_dir,
            start=start_date,
            init_window=init_window_size, 
            api_sleep=api_sleep,
        )

    (cache_dir / "BUILD_COMPLETE").touch()
    logger.info("Cache build complete: %s", cache_dir)


# ── reconstruction ─────────────────────────────────────────────────────────


def reconstruct_asof(
    target_version: int,
    entity_dir: Path,
    time_col: str,
) -> pd.DataFrame:
    """Reconstruct the time series as it appeared at *target_version*.

    Algorithm (preserved from ``construct.py:build_asof``):
    1. Start at target_version, read its delta file.
    2. From delta metadata, get the start index for this delta.
    3. Step back one position in the full index list.
    4. Look up which version last modified that previous time point.
    5. Jump to that version's delta — repeat from step 1.
    6. Stop when the delta's start <= base_max, or no metadata entry,
       or file missing, or lookup miss.
    7. Prepend the base series, concatenate, sort.
    """
    delta_dir = entity_dir / "deltas"

    base_series = pd.read_parquet(entity_dir / "base.parquet")
    with open(entity_dir / "delta_metadata.json") as f:
        delta_metadata: dict = json.load(f)
    with open(entity_dir / "max_version_lookup.json") as f:
        max_version_lookup: dict = json.load(f)

    # full index list for generic step-back (replaces hardcoded epiweek math)
    finalized_df = pd.read_parquet(entity_dir / "finalized.parquet")
    all_indices = finalized_df[time_col].sort_values().tolist()

    def step_back_one(current: int) -> int | None:
        try:
            idx = all_indices.index(current)
            return all_indices[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    start_key = f"start_{time_col}"
    available_versions = sorted(int(k) for k in delta_metadata)

    def _resolve_version(version: int) -> int | None:
        """Find *version* in delta_metadata, or fall back to the latest
        available version <= *version*.  Returns ``None`` when nothing
        qualifies."""
        if str(version) in delta_metadata:
            return version
        candidates = [v for v in available_versions if v <= version]
        if not candidates:
            return None
        fallback = candidates[-1]
        logger.info(
            "Version %d not in delta metadata, falling back to %d.",
            version, fallback,
        )
        return fallback

    result: list[pd.DataFrame] = []
    current_version = _resolve_version(target_version)
    current_start: int | None = None
    base_max = base_series[time_col].max() if not base_series.empty else float('-inf')

    while current_version is not None and \
            (current_start is None or current_start > base_max) and \
            str(current_version) in delta_metadata:

        file_path = delta_dir / f"asof={current_version}.parquet"
        if not file_path.exists():
            logger.warning("Delta file missing: %s. Stopping.", file_path)
            break

        delta = pd.read_parquet(file_path)

        if current_start is not None:
            delta = delta[delta[time_col] < current_start]

        result.append(delta)

        metadata = delta_metadata[str(current_version)]
        current_start = metadata[start_key]

        prev_time = step_back_one(current_start)  # type: ignore
        if prev_time is None:
            break

        # max_version_lookup may point beyond our build range;
        # resolve to the nearest available delta
        raw_version = max_version_lookup.get(str(prev_time))
        if raw_version is None:
            break
        current_version = _resolve_version(int(raw_version))

    if not base_series.empty:
        result.append(base_series)
    if not result:
        logger.warning(
            "No data found for version=%d in %s.", target_version, entity_dir,
        )
        return pd.DataFrame(columns=finalized_df.columns)
    return pd.concat(result).sort_values(time_col).reset_index(drop=True)
