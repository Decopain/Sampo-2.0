import pandas as pd
import numpy as np
from pathlib import Path
from sampo2.config import COMBINED_DATA_FILE, ACTION_MAP

class DataLoader:
    def __init__(self, filepath: Path = COMBINED_DATA_FILE):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Loads data from CSV, parses dates, sets index."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found at {self.filepath}")

        # Load
        df = pd.read_csv(self.filepath)
        
        # Check Time Column
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            df.sort_index(inplace=True)
        else:
            raise KeyError("CSV must contain 'Time' column.")

        # Basic Validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df
        return df

    def add_returns(self, df: pd.DataFrame, col='Close') -> pd.DataFrame:
        """Adds percentage returns to the dataframe."""
        df = df.copy()
        df['Returns'] = df[col].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resamples data to a specific timeframe.
        Aggregates OHLCV correctly.
        """
        if timeframe == '1D':
            rule = '1D'
        elif timeframe == '1H':
            rule = '1h'
        elif timeframe == '15min':
            rule = '15min'
        else:
            return df # Return as is if unknown

        # Define aggregation logic
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'volume': 'sum'
        }
        
        # Add other columns to agg dict (use 'last' or 'mean' as default)
        for c in df.columns:
            if c not in agg_dict:
                if pd.api.types.is_numeric_dtype(df[c]):
                    agg_dict[c] = 'mean' # Default for indicators
                else:
                    agg_dict[c] = 'last' # categorical

        resampled = df.resample(rule).agg(agg_dict)
        resampled.dropna(subset=['Close'], inplace=True) # Drop empty bins
        
        return resampled
