import pandas as pd
import numpy as np
from sampo2.data.loader import DataLoader
from sampo2.data.garch import GarchVolatilityModel
from sampo2.config import OUTPUT_DIR, COMBINED_DATA_FILE

class DataProcessor:
    def __init__(self):
        self.loader = DataLoader()
        self.garch = GarchVolatilityModel()

    def build_combined_dataset(self, 
                               base_data_path=COMBINED_DATA_FILE, 
                               enriched_features_path=None, 
                               output_csv_name="final_combined_dataset.csv"):
        
        print("--- Building Combined Dataset ---")
        
        # 1. Load Base Data
        print("Loading Base Data...")
        # Use our loader which handles parsing/indexing
        self.loader.filepath = base_data_path
        df = self.loader.load_data()
        df = self.loader.add_returns(df)
        
        # 2. Generate GARCH Features (on the fly to ensure usage)
        print("Generating GARCH Features...")
        # Fit on Hourly (since base is Hourly)
        # Note: In production we might load pre-computed, but calculating ensures data consistency
        vol_h1 = self.garch.fit_predict(df['Returns'])
        df['volatility_h1'] = vol_h1
        
        # 3. Merge Enriched Features
        if enriched_features_path and enriched_features_path.exists():
            print(f"Merging Enriched Features from {enriched_features_path}...")
            enriched_df = pd.read_csv(enriched_features_path)
            
            # Ensure index alignment
            if 'Time' in enriched_df.columns:
                enriched_df['Time'] = pd.to_datetime(enriched_df['Time'])
                enriched_df.set_index('Time', inplace=True)
            
            # Merge
            initial_len = len(df)
            df = df.join(enriched_df, how='left')
            
            # Fill NaNs from merge (maybe enriched features cover less range)
            df.fillna(method='ffill', inplace=True)
            df.fillna(0.0, inplace=True) # Fill remaining starts
            
            print(f"Merged. Shape: {df.shape}. (Base: {initial_len})")
            
        else:
            print("No enriched features provided, skipping merge.")

        # 4. Add Action Column (Optional placeholder)
        if 'action' not in df.columns:
            df['action'] = 0 # HOLD_OUT default
            
        # 5. Final Cleanup
        # Ensure no NaNs before passing to Env
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"Warning: {null_counts} NaNs remaining. Filling with 0.")
            df.fillna(0.0, inplace=True)

        # Save
        if output_csv_name:
            out_path = OUTPUT_DIR / output_csv_name
            df.to_csv(out_path)
            print(f"Saved combined dataset to {out_path}")
            
        return df

if __name__ == "__main__":
    processor = DataProcessor()
    # Assume enriched features exist in outputs
    enriched_path = OUTPUT_DIR / "enriched_features.csv"
    processor.build_combined_dataset(enriched_features_path=enriched_path)
