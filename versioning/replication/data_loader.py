import pandas as pd
import json
from datetime import datetime
from delphi_epidata import Epidata
from replication.config import DELPHI_API_KEY, DATA_DIR

# --- DATA INITIALIZATION ---

def load_versioning_artifacts():
    """
    Load versioning artifacts (base, deltas metadata, max issue lookup).

    Call this function to reload artifacts after generating data for a new region.
    """
    base_series = pd.read_parquet(DATA_DIR / "base.parquet")

    with open(DATA_DIR / "delta_metadata.json", "r") as f:
        delta_metadata = json.load(f)

    with open(DATA_DIR / "max_issue_lookup.json", "r") as f:
        max_issue_lookup = json.load(f)

    return delta_metadata, max_issue_lookup, base_series

def load_ili_data(region_code="nat", start_epi=201650, end_epi=202352):
    """Load ILI data from Delphi API."""
    Epidata.auth = ("epidata", DELPHI_API_KEY) #type:ignore
    res = Epidata.fluview([region_code], Epidata.range(start_epi, end_epi))
    df = pd.DataFrame(res["epidata"]) #type:ignore
    df["date"] = df["epiweek"].apply(epiweek_to_date)
    df = df.sort_values("date")[["date", "wili"]].dropna(subset=["wili"])
    df["wili"] = df["wili"].astype(float)
    df = df.set_index("date").asfreq("W-MON")
    df["wili"] = df["wili"].interpolate(limit_direction="both")
    return df

# --- UTILITIES ---

def epiweek_to_date(epiweek_value):
    s = str(int(epiweek_value))
    year, week = int(s[:4]), int(s[4:])
    return datetime.strptime(f"{year} {week} 1", "%G %V %u")

def date_to_epiweek(date):
    """Convert datetime to epiweek format YYYYWW."""
    iso_calendar = date.isocalendar()
    year = iso_calendar[0]  # or iso_calendar.year in Python 3.9+
    week = iso_calendar[1]  # or iso_calendar.week in Python 3.9+
    return year * 100 + week

def clean_asof_df(df: pd.DataFrame):
    df["date"] = df["epiweek"].apply(epiweek_to_date)
    df = df.sort_values("date")[["date", "wili"]].dropna(subset=["wili"])
    df["wili"] = df["wili"].astype(float)
    df = df.set_index("date").asfreq("W-MON")
    df["wili"] = df["wili"].interpolate(limit_direction="both")
    return df

# --- MAIN FUNCTION ---

def build_asof_series(
    as_of_epiweek,
    delta_metadata=None,
    max_issue_lookup=None,
    base_series=None
):
    """
    Build time series as of a specific epiweek.

    If artifacts are not provided, they will be loaded from DATA_DIR.
    This allows the function to work with newly generated versioning data.

    Args:
        as_of_epiweek: The epiweek to reconstruct data as-of
        delta_metadata: Optional delta metadata dict (loads from disk if None)
        max_issue_lookup: Optional max issue lookup dict (loads from disk if None)
        base_series: Optional base series DataFrame (loads from disk if None)
    """
    from replication.versioning import build_asof

    # Load artifacts if not provided
    if delta_metadata is None or max_issue_lookup is None or base_series is None:
        delta_metadata, max_issue_lookup, base_series = load_versioning_artifacts()

    # Build as-of series using versioning logic
    asof_df = build_asof(
        as_of_epiweek,
        delta_metadata=delta_metadata,
        max_issue_lookup=max_issue_lookup,
        base_series=base_series
    )

    cleaned_df = clean_asof_df(asof_df)

    return cleaned_df
