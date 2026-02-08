from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def step_index(indices: list, current: int, k: int) -> int | None:
    """Move *k* steps forward (positive) or backward (negative) in an ordered
    index list.  Returns ``None`` when out of bounds or *current* is absent."""
    try:
        current_idx = indices.index(current)
        target_idx = current_idx + k

        if target_idx >= len(indices) or target_idx < 0:
            return None

        return indices[target_idx]
    except ValueError:
        logger.error("Index value %s not found in index list.", current)
        return None


def earliest_active_index(
    indices: list[int],
    max_version_lookup: dict[str, int],
    version: int,
    start: int,
) -> int | None:
    """Return the first time index (from *start* onward) whose maximum version
    exceeds *version*.  This identifies where revisions still exist beyond the
    current version — the optimisation that lets us skip unchanged history."""
    i0 = indices.index(start)
    for t in indices[i0:]:
        # Convert t to string for the lookup because JSON keys are strings
        if max_version_lookup[str(t)] > version:
            logger.info(
                "Version %d active at index=%s (max_version=%d).",
                version, t, max_version_lookup[str(t)],
            )
            return t
    return None


def check_missing_indices(
    indices: list[int],
    returned_indices: list[int],
    start: int,
    end: int,
) -> list[int]:
    """Compare *returned_indices* against the expected range and return any
    missing values."""
    start_idx = indices.index(start)
    end_idx = indices.index(end)
    expected = set(indices[start_idx : end_idx + 1])

    if set(returned_indices) != expected:
        missing = list(expected - set(returned_indices))
        logger.info(
            "Queried [%s, %s] returned [%s, %s].",
            start, end,
            min(returned_indices) if returned_indices else "N/A",
            max(returned_indices) if returned_indices else "N/A",
        )
        if missing:
            logger.warning("Missing indices for version=%s: %s", end, missing)
        return missing
    return []
