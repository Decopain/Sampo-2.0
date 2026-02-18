
import pandas as pd
import time
import os
from sampo2.data.ib_aggregator import IBAggregator

def fetch_year(year, instrument="EUR_USD", granularity="H1", output_dir="outputs"):
    """
    Fetches data for a single year using IBAggregator (Random Client ID logic included in aggregator default? 
    No, aggregator.py MAIN block has it. I need to replicate that randomization here).
    """
    import random
    client_id = random.randint(1000, 9999) # Use different range to be safe
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    print(f"Fetching {year} ({start_date} to {end_date}) with Client ID {client_id}...")
    
    # We use IBAggregator directly
    # Note: IBAggregator.get_historical_data arguments:
    # symbol, sec_type='FOREX', currency='USD', exchange='SMART', duration='1 D', bar_size='1 hour', what_to_show='MIDPOINT', end_date='', useRTH=True
    
    # Calculate duration string
    s_dt = pd.to_datetime(start_date)
    e_dt = pd.to_datetime(end_date)
    delta = e_dt - s_dt
    duration_str = f"{delta.days + 1} D"
    
    # Map granularity to bar_size
    bar_size_map = {
        "D": "1 day",
        "H1": "1 hour",
        "M15": "15 mins",
        "M5": "5 mins",
        "M1": "1 min"
    }
    bar_size_str = bar_size_map.get(granularity, "1 hour")
    
    # Assuming config is dict or object, IBAggregator takes host/port/client_id directly
    # Default host/port
    host = '127.0.0.1'
    port = 7497
    
    try:
        with IBAggregator(host=host, port=port, client_id=client_id) as ib:
            df = ib.get_historical_data(
                symbol=instrument, 
                duration=duration_str, 
                bar_size=bar_size_str,
                what_to_show='MIDPOINT',
                # End date for IBKR reqHistoricalData is the END of the period. 
                # If we want 2006-01-01 to 2006-12-31, we ask for data ending at 2006-12-31 23:59:59?
                # Actually reqHistoricalData endDateTime defaults to NOW if empty.
                # If we provide EndDateTime, it looks back 'duration'.
                # So if duration is 365 Days, and we set end_date='20061231 23:59:59', we get 2006.
                end_date=e_dt.strftime("%Y%m%d 23:59:59"),
                useRTH=True 
            )
            
            if df is not None and not df.empty:
                # Save chunk
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{instrument}_{granularity}_{year}.csv"
                path = os.path.join(output_dir, filename) if output_dir else filename
                df.to_csv(path)
                print(f"Saved {year} to {path} ({len(df)} rows)")
                return df
            else:
                print(f"No data returned for {year}")
                return None
                
    except Exception as e:
        print(f"Error fetching {year}: {e}")
        return None

def main():
    years = [2006, 2007, 2008, 2009, 2010]
    instrument = "EUR_USD"
    # System Architecture requires H1, Daily, and M15
    granularities = ["H1", "D", "M15"]
    output_dir = "outputs/phase9_raw"
    
    for gran in granularities:
        print(f"\n=== Fetching {gran} Data ===")
        all_dfs = []
        
        for year in years:
            df = fetch_year(year, instrument, gran, output_dir)
            if df is not None:
                all_dfs.append(df)
            
            # Sleep to be polite to IBKR pacing
            print("Sleeping 5s...")
            time.sleep(5)
            
        if all_dfs:
            print(f"Concatenating all years for {gran}...")
            full_df = pd.concat(all_dfs)
            # Sort just in case
            full_df['Time'] = pd.to_datetime(full_df['Time'])
            full_df.sort_values('Time', inplace=True)
            
            final_path = os.path.join(output_dir, f"{instrument}_{gran}_2006_2010_merged.csv")
            full_df.to_csv(final_path, index=False)
            print(f"Complete dataset saved to: {final_path} ({len(full_df)} rows)")
        else:
            print(f"No data collected for {gran}.")

if __name__ == "__main__":
    main()
