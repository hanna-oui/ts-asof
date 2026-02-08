from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):
    """Interface that any versioned data source must satisfy.

    Must have two column names:
        1.  ``time_col`` (observation time) and,
        2. ``version_col`` (revision / issue identifier).
        3. (Optional) ``entity_col`` for panel data. 

    Furthermore, implementors must provide:
        1. ``fetch_snapshot`` (current / finalized data)
        2. ``fetch_revision`` (data as reported at a specific version).
        3. (Optional) ``cache_key_params`` (used to generate cache key for storing intermediate series)
    """

    @property
    @abstractmethod
    def time_col(self) -> str:
        """Observation time column name."""
        pass

    @property
    @abstractmethod
    def version_col(self) -> str:
        """Revision / issue identifier column name."""
        pass

    @property
    def entity_col(self) -> str | None:
        """Optional entity column for panel data (e.g. region). Defaults to None."""
        return None
    
    @property
    def entity_ids(self) -> list[str]:
        """Return the specific identifiers (e.g., ['nat', 'ca']) for this source."""
        return []

    @abstractmethod
    def fetch_finalized(self, start: int, end: int) -> pd.DataFrame:
        """Fetch the latest / finalized data for *start* … *end*."""
        pass

    @abstractmethod
    def fetch_revision(
        self, version: int, start: int, end: int,
    ) -> pd.DataFrame | None:
        """Fetch data as it was reported at *version* for the range *start* … *end*."""
        pass

    def cache_key_params(self) -> dict:
        """
        Return a JSON-serialisable dict of parameters for cache hashing.
        
        Default implementation: Captures all instance attributes that are 
        not private (start with _) and are JSON-serializable types.
        """
        allowed_types = (str, int, float, bool, list, dict, type(None))
        params = {}
        
        for k, v in self.__dict__.items():
            # Skip private attributes and internal machinery
            if k.startswith('_'):
                continue
            
            # Only include common JSON-serializable data types
            if isinstance(v, allowed_types):
                # Sort lists so that order doesn't change the hash
                params[k] = sorted(v) if isinstance(v, list) else v
                
        return params