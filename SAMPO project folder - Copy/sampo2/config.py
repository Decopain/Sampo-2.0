import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "SAMPO"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data Files (Raw)
DAILY_DATA_FILE = DATA_DIR / "2010-2011 D.csv"
H1_DATA_FILE = DATA_DIR / "2010-2011 H1.csv"
M15_DATA_FILE = DATA_DIR / "2010-2011 M15.csv"

# Combined (Processed)
COMBINED_DATA_FILE = DATA_DIR / "combined_data.csv"
ENRICHED_FEATURES_FILE = DATA_DIR / "enriched_features_df.csv"

# Timeframes
TIMEFRAMES = {
    "Daily": "1D",
    "H1": "1h",
    "15min": "15min"
}

# GARCH Config
GARCH_P = 1
GARCH_Q = 1
GARCH_MEAN = 'Zero'
RESCALE_FACTOR = 1000.0

# ACTION MAP
ACTION_MAP = {
    0: 'HOLD_OUT',
    1: 'BUY',
    2: 'SELL',
    3: 'CLOSE',
    4: 'HOLD_IN'
}
