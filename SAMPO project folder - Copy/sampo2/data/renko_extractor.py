
import pandas as pd
import numpy as np
from pathlib import Path
from sampo2.data.custom_indicators import RenkoIndicator
from sampo2.config import COMBINED_DATA_FILE, OUTPUT_DIR

# Dummy class to mimic TradeBar input expected by RenkoIndicator
class TradeBar:
    def __init__(self, time, open_, high, low, close):
        self.Time = time
        self.EndTime = time
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close

def generate_renko_dataset(input_file=COMBINED_DATA_FILE, block_size=0.0005, output_file=None):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.sort_values('Time', inplace=True)
    
    # Initialize Renko Indicator
    # block_points=50 (fixed points), as_percent=False
    renko = RenkoIndicator(name="Renko50", block_points=block_size, as_percent=False, store=10, instrument="EURUSD")
    
    renko_data = []
    
    print("Processing Renko blocks...")
    for idx, row in df.iterrows():
        # Create TradeBar input
        bar = TradeBar(row['Time'], row['Open'], row['High'], row['Low'], row['Close'])
        
        # Update Indicator
        # RenkoIndicator.Update returns True if a NEW block is formed or state changes significantly
        # However, checking the code, it returns True generally.
        # We need to capture the state *after* the update.
        renko.Update(bar)
        
        if renko.IsReady:
            state = renko.get_ObjectDictionary()
            # We only want to record when a *new block* is historically confirmed or creates a distinct state
            # But for simplicity, we can record the state at every bar and then filter for changes,
            # OR better: rely on the indicator's internal 'Value' changes.
            
            # The user wants to predict the *next* block. 
            # So we should record the Series of Renko States.
            
            # Include Standard Technical Features for ML Context
            data_row = {
                'Time': row['Time'],
                'Close': row['Close'],
                'Renko_Value': state['Ren_Value'], 
                'Blue': int(state['Blue']),
                'Red': int(state['Red']),
                'Yellow': int(state['Yellow']),
                'Blue_Yellow': int(state['Blue/Yellow']),
                'Red_Yellow': int(state['Red/Yellow']),
                'Flag': state['Flag']
            }
            
            # Add all other columns from source (SMA, ATR, volume, etc)
            for col in row.index:
                if col not in data_row and col != 'Time':
                    data_row[col] = row[col]
            
            renko_data.append(data_row)
            
    renko_df = pd.DataFrame(renko_data)
    
    # Filter to only rows where Renko Value CHANGED (i.e. a new block was printed)
    # This creates the "Brick-based" time series rather than "Time-based"
    renko_bricks = renko_df[renko_df['Renko_Value'].shift() != renko_df['Renko_Value']].copy()
    
    # Create Targets: What is the NEXT block's state?
    # We want to predict if the NEXT block (row + 1) will be Blue, Red, or Consolidation?
    # Actually, the user wants: "uptrend, downtrend, or elongate"
    
    # Create Multi-Class Target based on NEXT block's state
    # Classes:
    # 0 = Pure Red (Strong Down)
    # 1 = Pure Blue (Strong Up)
    # 2 = Red/Yellow (Stagnant Down)
    # 3 = Blue/Yellow (Stagnant Up)
    
    def get_class(row):
        is_blue = row['Blue'] == 1
        is_yellow = row['Yellow'] == 1
        
        if is_blue and not is_yellow:
            return 1 # Pure Blue
        elif not is_blue and not is_yellow:
            return 0 # Pure Red
        elif is_blue and is_yellow:
            return 3 # Blue/Yellow
        elif not is_blue and is_yellow:
            return 2 # Red/Yellow
        return 0 # Fallback
        
    # Apply to current row first to verify distribution
    renko_bricks['Class'] = renko_bricks.apply(get_class, axis=1)
    
    # Target is the Class of the NEXT block
    renko_bricks['Target_State'] = renko_bricks['Class'].shift(-1)
    
    # Drop last row (NaN target)
    renko_bricks.dropna(inplace=True)
    
    # Cast State to int
    renko_bricks['Target_State'] = renko_bricks['Target_State'].astype(int)
    
    if output_file:
         output_path = Path(output_file)
    else:
         output_path = OUTPUT_DIR / "renko_bricks_dataset.csv"

    renko_bricks.to_csv(output_path, index=False)

    print(f"Renko Dataset saved to {output_path} ({len(renko_bricks)} bricks)")
    
    # Print Class Distribution
    dist = renko_bricks['Target_State'].value_counts(normalize=True).sort_index()
    print("\nTarget Class Distribution:")
    classes = {0: "Pure Red", 1: "Pure Blue", 2: "Red/Yellow", 3: "Blue/Yellow"}
    for k, v in dist.items():
        print(f"{classes[k]}: {v:.1%}")

    return renko_bricks

if __name__ == "__main__":
    generate_renko_dataset()
