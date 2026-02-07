from pathlib import Path

# project root = TSFM/
ROOT = Path(__file__).resolve().parents[2]

SERIES = "wili"
REGIONS = ["nat"]

# date range (obs space)
START_EPIWEEK = 201650
END_EPIWEEK   = 202352


# rolling-origin params
INIT_TRAIN = 128
STEP       = 4
HORIZON    = 4

# paths
DATA_DIR  = ROOT / "research-log" / "versioning" / "data" / SERIES
DELTA_DIR = DATA_DIR / "deltas" 


# API
API_SLEEP = 1.0      # seconds between calls
MAX_RETRY = 5

