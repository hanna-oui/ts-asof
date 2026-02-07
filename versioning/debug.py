import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "research-log" / "nowcasting" / "data" / "wili"
DELTA_DIR = DATA_DIR / "deltas" 

file_path = DATA_DIR / 'base.parquet'

# Read the parquet file into a pandas DataFrame
df = pd.read_parquet(file_path, engine='pyarrow')

import code 
code.interact(local=locals())
