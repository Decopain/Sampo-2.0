
import pandas as pd
from pathlib import Path
import time
from sampo2.config import OUTPUT_DIR
from sampo2.data.ib_aggregator import IBAggregator

import logging

OUTPUT_FILE = OUTPUT_DIR / "history_2005_2009_ib.csv"

def fetch_ib_history():
    logging.basicConfig(level=logging.INFO)
    print("Initializing IB Aggregator...")
    # Port 7497 (Paper) or 7496 (Live). Defaulting to 7497 as per aggregator or TWS default.
    agg = IBAggregator(port=7497, client_id=104) 
    
    all_data = []
    
    try:
        # Loop from 2005 to 2009
        years = range(2005, 2010) # 2005..2009
        
        for year in years:
            target_end_str = f"{year+1}0101 00:00:00" # e.g., 20020101 gets all of 2001
            print(f"Fetching Year {year} (End Date: {target_end_str})...")
            
            df = agg.get_historical_data(
                symbol='EURUSD', # ib_aggregator handles logic
                sec_type='FOREX',
                duration='1 Y',
                bar_size='1 hour',
                end_date=target_end_str
            )
            
            if not df.empty:
                print(f"  Received {len(df)} rows.")
                all_data.append(df)
            else:
                print(f"  No data for {year}.")
            
            # Pause to avoid pacing violations
            time.sleep(2)
            
    except Exception as e:
        print(f"Global Error: {e}")
    finally:
        agg.disconnect()
        
    if all_data:
        full_df = pd.concat(all_data)
        full_df.sort_values('Time', inplace=True)
        full_df.drop_duplicates(subset=['Time'], inplace=True)
        full_df.reset_index(drop=True, inplace=True)
        
        print(f"Total Rows: {len(full_df)}")
        print(f"Range: {full_df['Time'].min()} to {full_df['Time'].max()}")
        
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    fetch_ib_history()
