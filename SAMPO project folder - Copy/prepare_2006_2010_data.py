
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add root to sys.path
sys.path.append(os.getcwd())

from sampo2.data.aggregator import SampoDataAggregator
from sampo2.data.renko_extractor import generate_renko_dataset
from sampo2.models.train_renko_xgboost import predict_full_dataset

# Paths
# Input is the raw merged H1 file
RAW_FILE = "outputs/phase9_raw/EUR_USD_H1_2006_2010_merged.csv"

# Intermediate: The raw file enriched with SMA/ATR/Vol etc.
ENRICHED_H1_FILE = "outputs/phase9_raw/EUR_USD_H1_2006_2010_enriched.csv"

# Intermediate: The Renko bricks generated from the ENRICHED file
BRICKS_FILE = "outputs/renko_bricks_2006_2010.csv"

# Final: The H1 enriched file with predictions added
FINAL_FILE = "outputs/tuning_dataset_2006_2010.csv"

def main():
    print(f"--- Preparing 2006-2010 Tuning Dataset (Corrected V2) ---")
    
    # 0. Feature Engineering (Aggregator)
    print(f"\n[1/4] enriching raw data with Technical Indicators...")
    if not Path(ENRICHED_H1_FILE).exists() or True: # Force run for now
        # Load raw data
        raw_df = pd.read_csv(RAW_FILE)
        
        # Init Aggregator
        agg = SampoDataAggregator() 
        
        # Run Pipeline (SMA, ATR, etc)
        enriched_df = agg.run(
            instrument="EUR_USD", 
            granularity="H1", 
            raw_df=raw_df,
            sma_s=5, sma_c=2, sma_l=12
        )
        
        # Add PNN placeholder columns (XGBoost expects pnn_0...pnn_99)
        # We fill with 0 since we don't have the PNN output ready for this specific range yet,
        # and checking the columns in bricks file for reference shows pnn_0...pnn_99
        print("Adding placeholder PNN features...")
        for i in range(100):
            enriched_df[f'pnn_{i}'] = 0.0
            
        # Add volatility_h1 (seen in bricks file)
        enriched_df['volatility_h1'] = enriched_df['returns'].rolling(24).std().fillna(0)

        # Save enriched H1
        enriched_df.to_csv(ENRICHED_H1_FILE)
        print(f"Saved Enriched H1 to {ENRICHED_H1_FILE}")
    else:
        print(f"Enriched H1 already exists at {ENRICHED_H1_FILE}")

    # 1. Generate Bricks (from Enriched Data so bricks have features!)
    print(f"\n[2/4] Generating Renko Bricks from {ENRICHED_H1_FILE}...")
    generate_renko_dataset(input_file=ENRICHED_H1_FILE, output_file=BRICKS_FILE)
    
    # 2. Run Inference
    print(f"\n[3/4] Running Renko XGBoost Inference...")
    # predict_full_dataset returns the merged dataframe
    df_merged = predict_full_dataset(input_csv=ENRICHED_H1_FILE, renko_file=BRICKS_FILE)
    
    # 3. Save Final Dataset
    print(f"\n[4/4] Saving Final Dataset to {FINAL_FILE}...")
    df_merged.to_csv(FINAL_FILE)
    print("Done!")

if __name__ == "__main__":
    main()
