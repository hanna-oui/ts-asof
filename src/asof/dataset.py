from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd

from asof.builder import build_cache, reconstruct_asof
from asof.sources.base import DataSource

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "asof"


class Dataset:
    """A versioned time series dataset that can reconstruct as-of views.

    Args:
        source: A `DataSource` implementation (e.g.
            Epidata). 
        start: First time index value (inclusive).
        end: Last time index value (inclusive).
        cache_dir: Explicit directory path for storing intermediate files.  
            When provided the cache is persistent (never auto-deleted).  
            A hash-based subdirectory is created inside it automatically.
        persist: If *True* (and *cache_dir* is not set), store the cache
            under ``~/.cache/asof/`` so it survives across sessions.
            Default is *False* — cache lives in a temp directory and is
            cleaned up when the ``Dataset`` object is garbage-collected
            or the interpreter exits.
        api_sleep: Seconds between API calls during cache build.
    """

    def __init__(
        self,
        data_source: DataSource,
        start_date: int, # very naive, should convert to date time
        end_date: int, # same as above
        init_window_size: int = 20,
        cache_dir: str | Path | None = None,
        persist: bool = False,
        api_sleep: float = 1.0,
    ) -> None:
        
        if not isinstance(data_source, DataSource):
            raise TypeError(
                f"source must implement DataSource, got {type(data_source).__name__}"
            )
        self._source = data_source
        self._start = start_date
        self._end = end_date
        self._init_window = init_window_size
        self._api_sleep = api_sleep

        self._temp_dir: tempfile.TemporaryDirectory | None = None

        if cache_dir:
            root = Path(cache_dir)
        elif persist and not cache_dir:
            root = _DEFAULT_CACHE_ROOT
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="asof_")
            root = Path(self._temp_dir.name)

        self._cache_dir = root / self._cache_key()
        self._built = self._build_completed()
        if not self._built:
            self.build()

    # ── cache management ──────────────────────────────────────────────

    def _cache_key(self) -> str:
        params = {
            **self._source.cache_key_params(),
            "start_date": self._start,
            "end_date": self._end,
            "init_window_size": self._init_window
        }
        raw = json.dumps(params, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _build_completed(self) -> bool:
        return (self._cache_dir / "BUILD_COMPLETE").exists()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def is_built(self) -> bool:
        return self._built

    # ── build ─────────────────────────────────────────────────────────

    def build(self) -> None:
        """Build the delta cache by fetching from the source.

        Robust to repeated calls, since existing delta files are saved. 
        """
        logger.info("Building dataset cache at %s", self._cache_dir)
        build_cache(
            source=self._source,
            start_date=self._start,
            end_date=self._end,
            init_window_size=self._init_window, 
            cache_dir=self._cache_dir,
            api_sleep=self._api_sleep,
        )
        self._built = True

    # ── entity helpers ────────────────────────────────────────────────

    @property
    def _is_multi_entity(self) -> bool:
        return self._source.entity_col is not None and len(self._source.entity_ids) > 1

    def _entity_dir(self, entity: str | None) -> Path:
        if not self._is_multi_entity:
            return self._cache_dir
        if entity is None:
            raise ValueError(
                "This dataset has multiple entities "
                f"(entity_col={self._source.entity_col!r}). Pass entity= to .asof()."
            )
        d = self._cache_dir / "entities" / str(entity)
        if not d.exists():
            raise ValueError(
                f"Entity {entity!r} not found in cache. "
                f"Available: {self.entities}"
            )
        return d

    @property
    def entities(self) -> list[str] | None:
        """Entity values present in the cache, or *None* for single-entity
        datasets."""
        if not self._is_multi_entity:
            return None
        if not self._built:
            raise ValueError("Cache not built. Call .build() or .asof() first.")
        entities_root = self._cache_dir / "entities"
        return sorted(d.name for d in entities_root.iterdir() if d.is_dir())

    # ── versions ──────────────────────────────────────────────────────

    @property
    def versions(self) -> list[int]:
        """Available versions from the delta metadata."""
        if not self._built:
            raise ValueError("Cache not built. Call .build() or .asof() first.")
        # read from first available entity dir
        if not self._is_multi_entity:
            meta_path = self._cache_dir / "delta_metadata.json"
        else:
            first_entity = self.entities[0]  # type: ignore[index]
            meta_path = (
                self._cache_dir / "entities" / first_entity / "delta_metadata.json"
            )
        with open(meta_path) as f:
            metadata = json.load(f)
        return sorted(int(k) for k in metadata)

    # ── asof reconstruction ───────────────────────────────────────────

    def asof(
        self,
        version: int,
        entity: str | None = None,
    ) -> pd.DataFrame:
        """Reconstruct the series as it appeared at *version*.

        For multi-entity (i.e., regions for epidemic data) datasets, 
        If *entity* is not specified, treat all as a single entity.
        If *entity* is ``None`` on a multi-entity dataset, all entities are
        returned concatenated with the entity column preserved.
        """
        tc = self._source.time_col

        # single entity or explicit entity
        if not self._is_multi_entity or entity is not None:
            return reconstruct_asof(
                target_version=version,
                entity_dir=self._entity_dir(entity),
                time_col=tc,
            )

        # multi-entity, no entity specified -> concatenate all
        frames = []
        for ent in self.entities:  # type: ignore[union-attr]
            df = reconstruct_asof(
                target_version=version,
                entity_dir=self._entity_dir(ent),
                time_col=tc,
            )
            frames.append(df)
        return pd.concat(frames).sort_values(tc).reset_index(drop=True)
