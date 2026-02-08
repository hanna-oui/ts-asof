# asof

Reconstruct versioned time series as they appeared at any point in time.

## Install

```bash
uv add asof
# With Epidata support:
uv add asof[epidata]
```

## Usage

```python
from asof import Dataset
from asof.sources import EpidataSource

ds = Dataset(
    source=EpidataSource(series="wili", regions=["nat"]),
    start=201650,
    end=202352,
)

# Reconstruct the series as it was known at epiweek 202001
df = ds.asof(202001)
```

The first `.asof()` call builds a local delta cache by fetching from the data source. Subsequent calls reconstruct instantly from cache.

## Panel data

```python
ds = Dataset(
    source=EpidataSource(series="wili", regions=["nat", "ca", "tx"]),
    start=201650,
    end=202352,
)

df_nat = ds.asof(202001, entity="nat")   # single region
df_all = ds.asof(202001)                  # all regions
```

## Custom data sources

Implement the `DataSource` protocol to support any revisable data:

```python
from asof.sources.base import DataSource

class MySource:
    @property
    def time_col(self) -> str:
        return "date"

    @property
    def version_col(self) -> str:
        return "revision"

    @property
    def entity_col(self) -> str | None:
        return None

    def fetch_snapshot(self, start, end):
        ...  # return DataFrame with time_col and version_col

    def fetch_revision(self, version, start, end):
        ...  # return DataFrame or None

    def cache_key_params(self) -> dict:
        return {"source": "my_source"}
```
