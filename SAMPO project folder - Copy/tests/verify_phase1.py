
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sampo2.data.loader import DataLoader
from sampo2.data.garch import GarchVolatilityModel
from sampo2.config import COMBINED_DATA_FILE

def test_phase1():
    print("=== Starting Phase 1 Verification ===")
    
    # 1. Test Loader
    print(f"Loading data from {COMBINED_DATA_FILE}...")
    loader = DataLoader()
    try:
        df = loader.load_data()
        print(f"Data Loaded Successfully. Shape: {df.shape}")
        print(f"Index: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"FAILED to load data: {e}")
        return

    # 2. Test Resampling
    print("\nTesting Resampling...")
    df_daily = loader.resample_data(df, '1D')
    print(f"Daily Shape: {df_daily.shape}")
    
    # 3. Test GARCH
    print("\nTesting GARCH Model...")
    garch = GarchVolatilityModel()
    
    # Fit Daily
    df_daily = loader.add_returns(df_daily)
    vol_daily = garch.fit_predict(df_daily['Returns'])
    print(f"Daily Volatility Generated. Mean: {vol_daily.mean():.6f}")
    
    # Fit Hourly (Base)
    df = loader.add_returns(df)
    vol_hourly = garch.fit_predict(df['Returns'])
    print(f"Hourly Volatility Generated. Mean: {vol_hourly.mean():.6f}")

    # Check for NaNs
    if vol_daily.isna().any():
        print("WARNING: NaNs found in Daily Volatility")
    else:
        print("Daily Volatility clean.")
        
    print("\n=== Phase 1 Verification Complete ===")

if __name__ == "__main__":
    test_phase1()
