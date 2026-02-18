
import tpqoa
import pandas as pd
from pathlib import Path
from sampo2.config import OUTPUT_DIR

OANDA_CFG = Path("c:/Users/Jeff/Desktop/sample_DS/oanda.cfg")
OUTPUT_FILE = OUTPUT_DIR / "history_2001_2009.csv"

def fetch_data():
    print(f"Connecting to Oanda using {OANDA_CFG}...")
    try:
        api = tpqoa.tpqoa(str(OANDA_CFG))
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    instrument = "EUR_USD"
    start = "2001-01-01"
    end = "2009-12-31"
    granularity = "H1"
    price = "M"

    print(f"Fetching {instrument} {granularity} from {start} to {end}...")
    try:
        # tpqoa handles pagination automatically
        df = api.get_history(instrument=instrument, start=start, end=end, granularity=granularity, price=price)
        
        # Reset index to make Time a column
        df.reset_index(inplace=True)
        # Rename 'time' to 'Time' if needed (tpqoa usually returns 'time' in index)
        # Columns usually: o, h, l, c, volume, complete
        # We need to map to: Time, Open, High, Low, Close, volume
        
        rename_map = {
            'time': 'Time',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'vol': 'volume' # tpqoa might return 'volume' or 'vol'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure title case columns
        df.rename(columns={'time': 'Time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        
        print(f"Downloaded {len(df)} rows.")
        if len(df) > 0:
            print(f"Sample:\n{df.head()}")
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved to {OUTPUT_FILE}")
        else:
            print("Warning: Downloaded empty dataset.")
            
    except Exception as e:
        print(f"Data download failed: {e}")

if __name__ == "__main__":
    fetch_data()
