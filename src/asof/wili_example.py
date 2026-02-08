from asof import Dataset
from asof.sources import DelphiEpidata

# initialize the dataset 
wili_data = Dataset(
    data_source=DelphiEpidata(
        series="wili",
        regions=["nat"],
        api_key="94c47e7d24c87",  # or set DELPHI_API_KEY env var and omit this
    ),
    start_date=201650,
    end_date=201901,
    init_window_size=100,
    persist=True
)

# First call builds the cache (fetches from API), subsequent calls are instant
df = wili_data.asof(201852)
