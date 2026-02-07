from pathlib import Path
import os
from dotenv import load_dotenv
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

# --- PATHS CONFIGURATION ---
# This identifies the directory where config.py actually lives
_current_file = Path(__file__).resolve()

# Logic: If config.py is inside 'replication/', root is one level up. 
# If it's in the root already, root is current parent.
if _current_file.parent.name == "replication":
    PROJECT_ROOT = _current_file.parent.parent
else:
    PROJECT_ROOT = _current_file.parent

SERIES = "wili"
DATA_DIR = PROJECT_ROOT / "data" / SERIES
# Fixed the 'replcation' typo here to match your main.py expectations
OUTPUT_DIR = PROJECT_ROOT / "replication" / "output"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------

# API
DELPHI_API_KEY = os.getenv('DELPHI_API_KEY')

# Model settings
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")
np.random.seed(12345)

# Suppress HuggingFace progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Plotting
C0 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]  # TimesFM
C1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]  # ARIMA
PALETTE = {"TimesFM": C0, "ARIMA": C1}
HUE_ORDER = ["TimesFM", "ARIMA"]

# CV Parameters
INIT_WINDOW = 128
STEP = 4
HORIZON = 4

# Regions
STATES = [
    'nat', 'al', 'az', 'ar', 'ca', 'de', 'ga', 'hi', 'id', 'il', 'ks', 'la', 'me', 'mn', 'ms',
    'mo', 'nv', 'nj', 'nc', 'oh', 'ok', 'pa', 'sc', 'tn', 'tx', 'va', 'wv', 'wi', 'wy'
]