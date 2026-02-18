
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tensorflow as tf

from sampo2.config import OUTPUT_DIR
from sampo2.data.aggregator import SampoDataAggregator
from sampo2.data.renko_extractor import generate_renko_dataset
from sampo2.models.pnn import ParallelNeuralNetwork
from sampo2.models.train_renko_xgboost import train_predictor
from sampo2.data.garch import GarchVolatilityModel # ARCHITECTURE COMPLIANCE
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONSTANTS ---
# Phase 9 Data (2006-2010, missing 2008)
RAW_FILE_PHASE9 = OUTPUT_DIR / "phase9_raw/EUR_USD_H1_2006_2010_merged.csv"
# Legacy files (commented out for clarity)
# RAW_FILE_2005 = OUTPUT_DIR / "history_2005_2009_ib.csv"
# FILE_2010 = Path("c:/Users/Jeff/Desktop/sample_DS/new new/2010-2011 H1.csv")
# FILE_2022 = Path("c:/Users/Jeff/Desktop/sample_DS/SAMPO/combined_data.csv")

PROCESSED_H1 = OUTPUT_DIR / "processed_h1.csv"
PROCESSED_DAILY = OUTPUT_DIR / "processed_daily.csv"
PROCESSED_M15 = OUTPUT_DIR / "processed_m15.csv"
PNN_MODEL_PATH = OUTPUT_DIR / "pnn_model_full.keras"
FINAL_COMBINED_FILE = OUTPUT_DIR / "final_combined_dataset.csv"

# --- 1. DATA INGESTION & AGGREGATION ---
def step_1_aggregate_data():
    # Check for Existing Processed Files
    if PROCESSED_H1.exists() and PROCESSED_DAILY.exists() and PROCESSED_M15.exists():
        print("Loading existing aggregated files...")
        return pd.read_csv(PROCESSED_H1, index_col='Time', parse_dates=True), \
               pd.read_csv(PROCESSED_DAILY, index_col='Time', parse_dates=True), \
               pd.read_csv(PROCESSED_M15, index_col='Time', parse_dates=True)

    print("\n[Step 1] Aggregating Raw Data & Feature Engineering...")
    
    def load_raw(path):
        if not path.exists(): return None
        df = pd.read_csv(path)
        rename_map = {'time':'Time', 'date':'Time', 'Date':'Time', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'vol':'volume'}
        df.rename(columns=rename_map, inplace=True)
        print(f"DEBUG: Loaded {path} raw shape: {df.shape}")
        if 'Time' in df.columns:
            # utc=True and format='mixed' robustly handles the timezone differences between IB and older files
            df['Time'] = pd.to_datetime(df['Time'], utc=True, format='mixed')
        # Normalize volume to exist
        return df[['Time', 'Open', 'High', 'Low', 'Close', 'volume']] if 'volume' in df.columns else df[['Time', 'Open', 'High', 'Low', 'Close']].assign(volume=0)

    dfs = [load_raw(RAW_FILE_PHASE9)]
    dfs = [d for d in dfs if d is not None]
    if not dfs: raise ValueError(f"No data found at {RAW_FILE_PHASE9}!")
    
    # Base H1
    unified = pd.concat(dfs).sort_values('Time').drop_duplicates('Time').reset_index(drop=True)
    
    print(f"DEBUG: Unified Raw Data Shape: {unified.shape}")
    print(f"DEBUG: Unified Columns: {unified.columns.tolist()}")
    print(f"DEBUG: Head:\n{unified.head()}")
    
    agg = SampoDataAggregator(config="oanda.cfg")
    
    # 1. H1 FEATURES
    processed_h1 = agg.run(instrument="EUR_USD", granularity="H1", raw_df=unified)
    processed_h1.to_csv(PROCESSED_H1, index=True)
    
    # 2. DAILY FEATURES (Resample)
    unified.set_index('Time', inplace=True)
    daily_raw = unified.resample('1D').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'volume':'sum'}).dropna().reset_index()
    processed_daily = agg.run(instrument="EUR_USD", granularity="D", raw_df=daily_raw)
    processed_daily.to_csv(PROCESSED_DAILY, index=True)
    
    # 3. M15 FEATURES (Resample/Upsample from H1)
    # Architecture requires M15. Since we don't have 18 years of M15 history downloaded,
    # we simulate the structure by upsampling H1. 
    # In Live environment, real M15 would be fed. This ensures matching shapes.
    unified_m15_raw = unified.resample('15min').ffill().reset_index()
    processed_m15 = agg.run(instrument="EUR_USD", granularity="M15", raw_df=unified_m15_raw)
    processed_m15.to_csv(PROCESSED_M15, index=True)
    
    print(f"Aggregations Complete: H1({len(processed_h1)}), D({len(processed_daily)}), M15({len(processed_m15)})")
    
    return processed_h1, processed_daily, processed_m15

# --- 2. GARCH ---
def step_2_garch_volatility(df_h1):
    print("\n[Step 2] GARCH Volatility Modeling...")
    garch = GarchVolatilityModel()
    if 'returns' not in df_h1.columns: df_h1['returns'] = df_h1['Close'].pct_change()
    returns = df_h1['returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_h1['volatility_h1'] = garch.fit_predict(returns)
    return df_h1

# --- 3. PNN TRAINING (3-Branch) ---
def step_3_train_pnn(df_h1, df_d, df_m15):
    print("\n[Step 3] Training PNN (M15 + H1 + Daily)...")
    
    # Align ALL to H1 Index (Renko base)
    # Daily needs ffill
    print(f"Aligning D({df_d.shape}) and M15({df_m15.shape}) to H1({df_h1.shape})...")
    
    # 1. Align (Reindex)
    df_d_aligned = df_d.reindex(df_h1.index, method='ffill')
    df_m15_aligned = df_m15.reindex(df_h1.index, method='ffill')
    
    # 2. Select Common Numeric Columns
    numeric_cols = df_h1.select_dtypes(include=[np.number]).columns
    # Intersection
    common_cols = [c for c in numeric_cols if c in df_d_aligned.columns and c in df_m15_aligned.columns]
    print(f"Common Columns: {len(common_cols)}")
    
    # 3. Extract Values & Cast to Float32
    X_h1 = df_h1[common_cols].values.astype('float32')
    X_d = df_d_aligned[common_cols].fillna(0).values.astype('float32')
    X_m15 = df_m15_aligned[common_cols].fillna(0).values.astype('float32')
    
    # Norm
    X_h1 = (X_h1 - X_h1.mean(axis=0)) / (X_h1.std(axis=0) + 1e-8)
    X_d = (X_d - X_d.mean(axis=0)) / (X_d.std(axis=0) + 1e-8)
    X_m15 = (X_m15 - X_m15.mean(axis=0)) / (X_m15.std(axis=0) + 1e-8)
    
    y = (df_h1['Close'].pct_change().shift(-1) > 0).astype(int).values
    
    X_h1, X_d, X_m15, y = X_h1[:-1], X_d[:-1], X_m15[:-1], y[:-1]
    
    # Build 3-Branch PNN
    input_shapes = {
        'h1': (X_h1.shape[1],), 
        'daily': (X_d.shape[1],),
        'm15': (X_m15.shape[1],)
    }
    pnn = ParallelNeuralNetwork(input_shapes, output_dim=1)
    
    # Train (Fallback to Training Mode due to feature mismatch with restored model)
    print("Training 3-branch PNN on Phase 9 Data...")
    pnn.fit([X_h1, X_d, X_m15], y, epochs=5, batch_size=256)
    pnn.save(PNN_MODEL_PATH)
    
    extractor = pnn.get_feature_extractor()
    embeddings = extractor.predict([X_h1, X_d, X_m15])
    
    emb_df = pd.DataFrame(embeddings, columns=[f'pnn_{i}' for i in range(embeddings.shape[1])])
    emb_df.index = df_h1.index[:-1]
    
    return emb_df

# --- 4. MERGE ---
def step_4_processor_merge(df_h1, emb_df):
    print("\n[Step 4] Final Processing (Merging All Features)...")
    final_df = df_h1.join(emb_df, how='inner')
    final_df.to_csv(FINAL_COMBINED_FILE, index=True)
    return FINAL_COMBINED_FILE

def step_5_renko_xgboost(final_file):
    print("\n[Step 5] Renko Extraction & XGBoost Training...")
    # Generate Bricks
    generate_renko_dataset(input_file=final_file, block_size=0.0005)
    
    # Train New XGBoost (Matched to PNN)
    train_predictor()
    
    # ENRICH WITH PREDICTIONS
    from sampo2.models.train_renko_xgboost import predict_full_dataset
    from sampo2.models.train_renko_xgboost import predict_full_dataset
    df_enriched = predict_full_dataset(final_file)
    df_enriched.to_csv(final_file, index=True) # Overwrite with Enriched Data
    print(f"Saved Enriched Dataset with Renko Predictions to {final_file}")

if __name__ == "__main__":
    # 1. Aggregate (H1, D, M15)
    df_h1, df_d, df_m15 = step_1_aggregate_data()
    
    # 2. GARCH
    df_h1 = step_2_garch_volatility(df_h1)
    
    # 3. PNN (3 Branch)
    emb_df = step_3_train_pnn(df_h1, df_d, df_m15)
    
    # 4. Processor
    final_path = step_4_processor_merge(df_h1, emb_df)
    
    # 5. Renko
    step_5_renko_xgboost(final_path)
    
    print("\n[SUCCESS] Strict System Architecture (3-Timeframe) Completed.")
