
import tpqoa
import logging

logging.basicConfig(level=logging.INFO)

def validate_oanda_connection(config_file="oanda.cfg"):
    print(f"Testing OANDA connection using {config_file}...")
    try:
        api = tpqoa.tpqoa(config_file)
        account_summary = api.get_account_summary()
        print("\n✅ Connection Successful!")
        print(f"Account ID: {api.account_id}")
        print(f"Account Type: {api.account_type}")
        print(f"Balance: {account_summary['balance']}")
        print(f"NAV: {account_summary['NAV']}")
        
        # Test basic data fetch
        print("\nTesting Data Fetch (EUR_USD, daily, last 5 days)...")
        df = api.get_history(instrument="EUR_USD", start="2023-01-01", end="2023-01-05", granularity="D")
        print(f"Fetched {len(df)} rows.")
        print(df.head())
        
        return True
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    validate_oanda_connection()
