
import asyncio
import pandas as pd
from ib_async import IB, util, Forex, Stock, CFD, Contract
from datetime import datetime, timedelta
import logging
import traceback

class IBAggregator:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Connects to the IBKR Termial/Gateway."""
        if not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self.logger.info(f"Connected to IBKR at {self.host}:{self.port} with Client ID {self.client_id}")
            except Exception as e:
                self.logger.error(f"Failed to connect to IBKR: {e}")
                raise

    def disconnect(self):
        """Disconnects from IBKR."""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IBKR")

    def get_historical_data(self, symbol, sec_type='FOREX', currency='USD', exchange='SMART', 
                            duration='1 D', bar_size='1 hour', what_to_show='MIDPOINT', end_date='', useRTH=True):
        """
        Fetches historical data for a given instrument.
        
        Args:
            symbol (str): The symbol (e.g., 'EUR').
            sec_type (str): Security type ('FOREX', 'STK', 'CFD').
            currency (str): Currency (e.g., 'USD').
            exchange (str): Exchange (e.g., 'IDEALPRO' or 'SMART').
            duration (str): Duration string (e.g., '1 Y', '1 M', '1 D').
            bar_size (str): Bar size setting (e.g., '1 hour', '1 min', '1 day').
            what_to_show (str): Type of data ('MIDPOINT', 'TRADES', 'BID', 'ASK').
            end_date (str): End date/time for the request (empty string for 'now').
            
        Returns:
            pd.DataFrame: DataFrame with historical data (Open, High, Low, Close, Volume).
        """
        try:
            self.connect()
            
            # defined contract based on type
            if sec_type == 'FOREX':
                # Symbol for Forex in ib_async is usually the pair like 'EURUSD' passed to Forex constructor
                # But if symbol is 'EUR' and currency is 'USD', we construct it appropriately.
                # Assuming input symbol might be 'EUR_USD' from OANDA format, we merge or split.
                # Common IBKR Forex: symbol='EURUSD'
                if '_' in symbol:
                    # e.g., 'EUR_USD' -> 'EURUSD'
                    formatted_symbol = symbol.replace('_', '')
                    contract = Forex(formatted_symbol)
                elif len(symbol) == 6:
                    # e.g. 'EURUSD'
                    contract = Forex(symbol)
                else:
                    # e.g. 'EUR' + 'USD' -> 'EURUSD'
                    contract = Forex(symbol + currency)
            elif sec_type == 'STK':
                contract = Stock(symbol, exchange, currency)
            elif sec_type == 'CFD':
                contract = CFD(symbol, exchange, currency)
            else:
                self.logger.warning(f"Unknown sec_type {sec_type}, defaulting to Contract(symbol=symbol, ...)")
                contract = Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency=currency)

            # Qualify contract
            self.ib.qualifyContracts(contract)
            self.logger.info(f"Qualified contract: {contract}")

            # Request historical data
            # Request historical data
            # Notebook used useRTH=True. Large requests might need chunking, but let's try matching params first.
            self.logger.info(f"Requesting data for {contract} duration={duration} barSize={bar_size} RTH={useRTH}")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=useRTH,
                formatDate=1
            )

            if not bars:
                self.logger.warning("No data returned from IBKR.")
                return pd.DataFrame()

            df = util.df(bars)
            
            # Standardize columns to match project expectation (Open, High, Low, Close, Volume)
            # IB return lower case: date, open, high, low, close, volume, barCount, average
            df.rename(columns={
                'date': 'Time',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Ensure Time is datetime
            df['Time'] = pd.to_datetime(df['Time'])
            # Set index to Time just in case, or keep as column depending on aggregator needs.
            # OANDA aggregator usually expects Time as a column or index. Let's keep it as column for now unless required.
            
            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    # Context manager support
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

if __name__ == "__main__":
    # Simple test block
    logging.basicConfig(level=logging.INFO)
    from ib_async import util
    # It might be necessary to run util.startLoop() if running outside of notebook interaction loop context,
    # but Aggregator usually runs as a script. ib_async handles loop internally in 'connect'? 
    # Usually util.startLoop() is for notebooks. In scripts, IB().connect() is blocking or uses threaded loop. 
    # Let's verify specific ib_async usage for scripts. 
    # The notebook says: "util.startLoop()"
    # For scripts, usually:
    # ib = IB()
    # ib.connect(...)
    # ib.run()  <-- if main thread needs to be blocked
    # But here we are making synchronous-looking calls via ib_async wrapper which handles event loop.
    
    # Try fetching some data (assuming TWS/Gateway is running on local)
    agg = IBAggregator(port=7497, client_id=99)
    try:
        # Example for EURUSD
        print("Fetching EURUSD 1 Day data...")
        df = agg.get_historical_data('EUR_USD', duration='10 D', bar_size='1 day')
        print(df.head())
        print(df.tail())
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        agg.disconnect()
