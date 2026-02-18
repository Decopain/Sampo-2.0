
import pandas as pd
import numpy as np
from pathlib import Path
from sampo2.config import OUTPUT_DIR, COMBINED_DATA_FILE
from sampo2.data.aggregator import SampoDataAggregator
from sampo2.data.renko_extractor import generate_renko_dataset
from sampo2.models.train_renko_xgboost import train_predictor

# Paths
FILE_2010 = Path("c:/Users/Jeff/Desktop/sample_DS/new new/2010-2011 H1.csv")
FILE_2022 = Path("c:/Users/Jeff/Desktop/sample_DS/SAMPO/combined_data.csv")
FILE_2005 = OUTPUT_DIR / "history_2005_2009_ib.csv"

# Intermediate and Final Files
UNIFIED_RAW_FILE = OUTPUT_DIR / "unified_data_raw.csv"
UNIFIED_PROCESSED_FILE = OUTPUT_DIR / "unified_data_processed.csv"

def load_and_standardize(filepath):
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Standardize Column Names
    rename_map = {
        'time': 'Time', 'Date': 'Time', 'date': 'Time',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'vol': 'volume', 'Volume': 'volume'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure Time
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], utc=True, format='mixed')
    
    # Select only Raw OHLCV
    req_cols = ['Time', 'Open', 'High', 'Low', 'Close']
    available = [c for c in req_cols if c in df.columns]
    
    if len(available) < 5:
        # Check if we can infer volume
        if 'volume' in df.columns:
            available.append('volume')
        else:
            df['volume'] = 0 # Dummy volume
            available.append('volume')
            
    return df[available]

def merge_raw_datasets():
    print("Merging RAW datasets...")
    
    dfs = []
    for f in [FILE_2005, FILE_2010, FILE_2022]:
        df = load_and_standardize(f)
        if df is not None:
             print(f"Loaded {f.name}: {df.shape}")
             dfs.append(df)
    
    if not dfs:
        return None
        
    # Concatenate
    df_unified = pd.concat(dfs, axis=0)
    df_unified.sort_values('Time', inplace=True)
    df_unified.drop_duplicates(subset=['Time'], inplace=True)
    df_unified.reset_index(drop=True, inplace=True)
    
    print(f"Unified Raw Dataset: {df_unified.shape}")
    df_unified.to_csv(UNIFIED_RAW_FILE, index=False)
    return UNIFIED_RAW_FILE

def process_via_aggregator(raw_file):
    print("Processing via Architecture (Aggregator)...")
    df_raw = pd.read_csv(raw_file)
    
    # Initialize Aggregator
    # We pass a dummy config string as we are providing raw_df directly
    aggregator = SampoDataAggregator(config="oanda.cfg")
    
    # Run the pipeline (which calls custom_indicators, etc.)
    df_processed = aggregator.run(
        instrument="EUR_USD",
        granularity="H1",
        raw_df=df_raw
    )
    
    print(f"Processed Dataset: {df_processed.shape}")
    df_processed.to_csv(UNIFIED_PROCESSED_FILE, index=True) # Aggregator returns Time index
    return UNIFIED_PROCESSED_FILE

if __name__ == "__main__":
    # 1. Merge Raw
    raw_path = merge_raw_datasets()
    
    # 2. Process via Standard Architecture
    processed_path = process_via_aggregator(raw_path)
    
    # 3. Extract Renko Info (Now using the properly processed file)
    print("\n--- Generating Renko Bricks ---")
    generate_renko_dataset(input_file=processed_path, block_size=0.0005)
    
    # 4. Retrain Model
    print("\n--- Retraining XGBoost ---")
    train_predictor()
