"""asof — Reconstruct versioned time series as they appeared at any point in time.

::

    from asof import Dataset
    from asof.sources import EpidataSource

    ds = Dataset(
        source=EpidataSource(series="wili", regions=["nat"]),
        start=201650,
        end=202352,
    )
    df = ds.asof(202001)
"""

from asof.dataset import Dataset
from asof.sources.base import DataSource

__all__ = ["Dataset", "DataSource"]
__version__ = "0.1.0"
