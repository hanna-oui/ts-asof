from __future__ import annotations
from asof.sources import DataSource

import logging
import pandas as pd

logger = logging.getLogger(__name__)

_SERIES_MAP = {
    "fluview": "fluview",
    "wili" : "fluview"
}


class DelphiEpidata(DataSource):
    """DataSource implementation for the CMU Delphi Epidata API.

    Args:
        series: Epidata series name. Just fluview support right now
        regions: Region codes, e.g. ``["nat"]`` or ``["nat", "ca", "tx"]``.
        api_key: Delphi API key.  Falls back to the ``DELPHI_API_KEY``
            environment variable when *None*.
    """

    def __init__(
        self,
        series: str,
        regions: list[str],
        api_key: str | None,
    ) -> None:
        if series not in _SERIES_MAP:
            raise ValueError(
                f"Unknown series {series!r}. Supported: {list(_SERIES_MAP)}"
            )
        self._series = series
        self._regions = regions 
        if not self._regions: 
            raise ValueError(
                "Please specify a region for your query."
            )
        self._api_key = api_key 
        if not self._api_key:
            raise ValueError(
                "Delphi API key required. Pass api_key= or set DELPHI_API_KEY."
            )

        try:
            from delphi_epidata import Epidata
        except ImportError:
            raise ImportError(
                "delphi-epidata is required for EpidataSource. "
                "Install with: pip install asof[epidata]"
            ) from None

        self._Epidata = Epidata
        self._Epidata.auth = ("epidata", self._api_key)  # type: ignore[attr-defined]
        self._api_fn = getattr(Epidata, _SERIES_MAP[series])

    # -- DataSource protocol properties --

    @property
    def time_col(self) -> str:
        return "epiweek"

    @property
    def version_col(self) -> str:
        return "issue"

    @property
    def entity_col(self) -> str | None:
        return "region"  # Always return "region" because the API always provides it
    
    @property
    def entity_ids(self) -> list[str]:
        return self._regions

    # -- DataSource protocol methods --
    def fetch_finalized(self, start: int, end: int) -> pd.DataFrame:
        res = self._api_fn(
            self._regions,
            epiweeks=self._Epidata.range(start, end),
        )["epidata"]  # type: ignore[index]
        df = pd.DataFrame(res)
        return df

    def fetch_revision(
        self, version: int, start: int, end: int,
    ) -> pd.DataFrame | None:
        res = self._api_fn(
            self._regions,
            epiweeks=self._Epidata.range(start, end),
            issues=version,
        )["epidata"]  # type: ignore[index]

        if not res:
            logger.warning("Version %d returned no data.", version)
            return None

        df = pd.DataFrame(res)
        logger.info(
            "Revision: version=%d, %d rows, range=[%d, %d]",
            version, len(df), start, end,
        )
        return df

    def cache_key_params(self) -> dict:
        return {
            "source": "epidata",
            "series": self._series,
            "regions": sorted(self._regions),
        }
