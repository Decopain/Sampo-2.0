"""
Sampo Data Aggregation Module
-----------------------------
This module handles fetching data from OANDA brokerage and aggregating it
into the multi-timeframe format required by the SAMPO2 pipeline.

This is a modular port of the original "Sampo Data Agregation (1).py" script.
It uses custom_indicators.py for: Renko, Fractals, Heikin-Ashi, Volume indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.signal import argrelextrema
import logging
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# OANDA API
try:
    import tpqoa
    TPQOA_AVAILABLE = True
except ImportError:
    TPQOA_AVAILABLE = False
    logging.warning("tpqoa not installed. Install with: pip install tpqoa")

from sampo2.config import OUTPUT_DIR

# Import custom indicators (copied from SAMPO folder)
try:
    from sampo2.data import custom_indicators as ci
    CUSTOM_INDICATORS_AVAILABLE = True
except ImportError:
    CUSTOM_INDICATORS_AVAILABLE = False
    logging.warning("custom_indicators not found in sampo2/data/")


class SampoDataAggregator:
    """
    Full data aggregation pipeline matching the original Sampo Data Agregation (1).py.
    
    Fetches data from OANDA, applies all custom indicators, and outputs
    a feature-rich dataset ready for the SAMPO2 pipeline.
    
    Usage:
        aggregator = SampoDataAggregator(config_file="oanda.cfg")
        df = aggregator.run(
            instrument="EUR_USD",
            start="2010-01-01",
            end="2019-12-31",
            granularity="D"
        )
        aggregator.save(df, "EUR_USD_D_output.csv")
    """
    
    def __init__(self, config="oanda.cfg"):
        """
        Initialize the aggregator.
        
        Args:
            config: Path to oanda.cfg file (str) OR Config object for IBKR.
        """
        self.config = config
        self.api = None
        self.config_file = config if isinstance(config, str) else "oanda.cfg"
        
        # If config is a string (path) and TPQOA is available, try to init OANDA
        if isinstance(config, str) and TPQOA_AVAILABLE:
            try:
                self.api = tpqoa.tpqoa(config)
                logging.info(f"OANDA API initialized from {config}")
            except Exception as e:
                logging.warning(f"Could not initialize OANDA API: {e}")
    
    def fetch_raw_data(self, instrument, start, end, granularity):
        """
        Fetch raw bid/ask data from OANDA.
        
        Returns:
            DataFrame with OHLCV, spread, and Time columns
        """
        if not self.api:
            raise RuntimeError("OANDA API not initialized. Check config file.")
        
        ask = self.api.get_history(
            instrument=instrument, start=start, end=end,
            granularity=granularity, price="A"
        )
        bid = self.api.get_history(
            instrument=instrument, start=start, end=end,
            granularity=granularity, price="B"
        )
        
        # Process data
        bid['spread'] = ask['c'] - bid['c']
        bid.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close'}, inplace=True)
        bid['price'] = bid['Close']
        bid['Time'] = bid.index
        bid.index = pd.to_datetime(bid.index)
        
        return bid
    
    def add_sma_features(self, df, sma_s=5, sma_c=2, sma_l=12):
        """Add SMA indicators and direction signals."""
        df = df.copy()
        
        df["SMA_S"] = df["price"].rolling(sma_s).mean()
        df["SMA_C"] = df["price"].rolling(sma_c).mean()
        df["SMA_L"] = df["price"].rolling(sma_l).mean()
        
        # SMA direction combinations
        df["up_sma1"] = ((df["SMA_C"] > df["SMA_S"]) & (df["SMA_S"] > df["SMA_L"])).astype(int)
        df["up_sma2"] = ((df["SMA_C"] < df["SMA_S"]) & (df["SMA_S"] > df["SMA_L"])).astype(int)
        df["down_sma1"] = ((df["SMA_C"] > df["SMA_S"]) & (df["SMA_S"] < df["SMA_L"])).astype(int)
        df["down_sma2"] = ((df["SMA_C"] < df["SMA_S"]) & (df["SMA_S"] < df["SMA_L"])).astype(int)
        df["flat_sma1"] = ((df["SMA_S"] > df["SMA_L"]) & (df["SMA_L"] < df["SMA_C"])).astype(int)
        df["flat_sma2"] = ((df["SMA_S"] < df["SMA_L"]) & (df["SMA_L"] > df["SMA_C"])).astype(int)
        
        return df
    
    def add_volume_indicators(self, df):
        """Add OBV and VWAP locally."""
        if df is None or 'volume' not in df.columns:
            return None
        
        vol = df.copy()
        
        # OBV
        vol["OBV"] = (np.sign(vol["Close"].diff()) * vol["volume"]).fillna(0).cumsum()
        
        # VWAP
        volume = vol['volume']
        price = vol['Close']
        cum_vol = volume.cumsum()
        cum_pv = (volume * price).cumsum()
        
        # Safe division
        vol['VWAP'] = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
        # If VWAP is 0, replace with Close
        vol['VWAP'] = vol['VWAP'].mask(vol['VWAP'] == 0, vol['Close'])
        vol['VWAP'] = vol['VWAP'].ffill()
        
        return vol[['OBV', 'VWAP']].assign(Time=vol.index)
    
    def add_heikin_ashi(self, bid_df, instrument, start, end, granularity):
        """Add Heikin-Ashi indicators."""
        if not CUSTOM_INDICATORS_AVAILABLE:
            return None
            
        ha = ci.HeikinAshiBar(
            self.config_file, # This config_file is for OANDA, might not be relevant for HA if data is from IBKR
            instrument=instrument,
            start=start,
            end=end,
            granularity=granularity,
            price="A" # This price is OANDA specific
        )
        for ind, data in bid_df.iterrows():
            ha.Update(data)
        
        hi = pd.DataFrame(ha.research)
        # Keep only essential columns (Time, HI_Value, BearWarning, BullWarning)
        hi = hi.drop(hi.columns[[2, 3, 4, 5, 6, 7]], axis=1, errors='ignore')
        return hi
    
    def add_fractals(self, bid_df, instrument):
        """Add Williams Fractal indicators."""
        if not CUSTOM_INDICATORS_AVAILABLE:
            return None
            
        fb = ci.FractalBar('CustFractal', [10, 3], instrument=instrument)
        for ind, data in bid_df.iterrows():
            fb.Update(data)
        
        fr = pd.DataFrame(fb.research)
        return fr
    
    def add_renko(self, bid_df, instrument):
        """Add Renko bar indicators."""
        if not CUSTOM_INDICATORS_AVAILABLE:
            return None
            
        ren = ci.RenkoIndicator('MyRenko', 0.005, as_percent=False, instrument=instrument)
        for ind, data in bid_df.iterrows():
            ren.Update(data)
        
        rb = pd.DataFrame(ren.research)
        # Drop some columns as in original
        rb = rb.drop(rb.columns[[0, 1, 2, 3, 6, 13]], axis=1, errors='ignore')
        return rb
    
    def add_peaks_crusts(self, df, order=5):
        """Add local minima/maxima detection."""
        df = df.copy()
        df['peak'] = False
        df['crust'] = False
        
        ilocs_min = argrelextrema(df.Close.values, np.less_equal, order=order)[0]
        ilocs_max = argrelextrema(df.Close.values, np.greater_equal, order=order)[0]
        
        df.iloc[ilocs_min, df.columns.get_loc('crust')] = True
        df.iloc[ilocs_max, df.columns.get_loc('peak')] = True
        
        return df
    
    def add_atr(self, df, period=14):
        """Add Average True Range."""
        df = df.copy()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(period).sum() / period
        return df
    
    def add_derived_signals(self, df):
        """Add combined signals (Fractal+SMA, Heikin+SMA, etc.)."""
        df = df.copy()
        
        # Fractal + SMA signals
        if 'Flag' in df.columns:
            df["buy_Frac"] = ((df.Flag == 1.0) & (df["SMA_S"] > df["SMA_L"])).astype(int)
            df["sell_Frac"] = ((df.Flag == 1.0) & (df["SMA_S"] < df["SMA_L"])).astype(int)
            df["neg_buy_Frac"] = ((df.Flag == 2.0) & (df["SMA_S"] < df["SMA_L"])).astype(int)
            df["neg_sell_Frac"] = ((df.Flag == 2.0) & (df["SMA_S"] > df["SMA_L"])).astype(int)
        
        # Heikin-Ashi + SMA signals
        if 'BullWarning' in df.columns:
            df['hi_up'] = ((df.BullWarning) & (df["SMA_S"] > df["SMA_L"])).astype(int)
            df['hi_down'] = ((df.BearWarning) & (df["SMA_S"] < df["SMA_L"])).astype(int)
        
        # Returns and direction
        df["returns"] = np.log(df.Close / df.Close.shift())
        df["t_dir"] = np.where(df["returns"] > 0, 1, 0)
        
        return df
    
    def run(self, instrument="EUR_USD", start=None, end=None, granularity="H1", raw_df=None, use_ib=False, sma_s=5, sma_c=2, sma_l=12):
        """
        Run the full data aggregation pipeline.
        
        Args:
            instrument: Trading pair (e.g., "EUR_USD")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            granularity: OANDA granularity ("D", "H1", "M15")
            sma_s, sma_c, sma_l: SMA periods
            
        Returns:
            DataFrame with all features
        """
        logging.info(f"Starting aggregation for {instrument} ({granularity}) from {start} to {end}")
        
        # 1. Fetch raw data
        # 1. Fetch raw data
        if raw_df is not None:
            bid = raw_df.copy()
            # Ensure proper index
            if 'Time' in bid.columns:
                bid['Time'] = pd.to_datetime(bid['Time'])
                bid.set_index('Time', drop=False, inplace=True)
            if 'price' not in bid.columns:
                bid['price'] = bid['Close']
            bid['EndTime'] = bid.index
            logging.info(f"Using local raw data: {len(bid)} rows")
            
        elif use_ib:
            from sampo2.data.ib_aggregator import IBAggregator
            # Determine bar size and duration
            bar_size_map = {
                'M1': '1 min', 'M5': '5 mins', 'M15': '15 mins', 'M30': '30 mins',
                'H1': '1 hour', 'H4': '4 hours', 'D': '1 day'
            }
            bar_size = bar_size_map.get(granularity, '1 hour')
            
            duration_str = "1 M" 
            if start and end:
                try:
                    s_dt = pd.to_datetime(start)
                    e_dt = pd.to_datetime(end)
                    delta = e_dt - s_dt
                    if delta.days > 0:
                        duration_str = f"{delta.days + 1} D"
                except:
                    pass
            
            logging.info(f"Fetching IBKR data: {instrument}, {bar_size}, {duration_str}")
            
            host = getattr(self.config, 'IB_HOST', '127.0.0.1') if not isinstance(self.config, str) else '127.0.0.1'
            port = getattr(self.config, 'IB_PORT', 7497) if not isinstance(self.config, str) else 7497
            # Randomize Client ID to avoid conflicts with open notebooks (usually ID 1)
            import random
            random_id = random.randint(10, 999)
            client_id = getattr(self.config, 'IB_CLIENT_ID', random_id) if not isinstance(self.config, str) else random_id
            
            with IBAggregator(host=host, port=port, client_id=client_id) as ib:
                bid = ib.get_historical_data(
                    symbol=instrument, 
                    sec_type='FOREX', 
                    duration=duration_str,
                    bar_size=bar_size
                )
            
            if bid.empty:
                raise ValueError("IBKR returned no data.")
                
            # Prepare format for indicators (needs 'Time' index, 'price' col)
            bid.set_index('Time', drop=False, inplace=True)
            bid['price'] = bid['Close']
            bid['EndTime'] = bid.index
            # Dummy spread to match structure if needed
            bid['spread'] = 0.0001 
            
            logging.info(f"Fetched {len(bid)} rows from IBKR.")

        else:
            # OANDA fallback
            bid = self.fetch_raw_data(instrument, start, end, granularity)
            bid['EndTime'] = bid.index
            logging.info(f"Fetched {len(bid)} candles from OANDA")
        
        # 2. Add SMA features
        sma_df = self.add_sma_features(bid, sma_s, sma_c, sma_l)
        sma_df = sma_df.drop('price', axis=1, errors='ignore')
        
        # 3. Add Volume indicators (OBV, VWAP)
        vol = self.add_volume_indicators(bid)
        
        # 4. Add Heikin-Ashi
        hi = self.add_heikin_ashi(bid, instrument, start, end, granularity)
        
        # 5. Add Fractals
        fr = self.add_fractals(bid, instrument)
        
        # 6. Add Renko
        rb = self.add_renko(bid, instrument)
        
        # 7. Join all features
        forecast = sma_df.copy()
        
        if rb is not None and 'Time' in rb.columns:
            forecast = forecast.set_index('Time').join(rb.set_index('Time'), rsuffix='_rb')
        if fr is not None and 'Time' in fr.columns:
            forecast = forecast.join(fr.set_index('Time'), rsuffix='_fr')
        if hi is not None and 'Time' in hi.columns:
            forecast = forecast.join(hi.set_index('Time'), rsuffix='_hi')
        if vol is not None and 'Time' in vol.columns:
            forecast = forecast.join(vol.set_index('Time'), rsuffix='_vol')
        
        # Reset index
        forecast = forecast.reset_index()
        forecast = forecast.rename(columns={'index': 'Time'})
        forecast = forecast.set_index('Time')
        
        # 8. Add peaks/crusts
        forecast = self.add_peaks_crusts(forecast)
        
        # 9. Add ATR
        forecast = self.add_atr(forecast)
        
        # 10. Add derived signals
        forecast = self.add_derived_signals(forecast)
        
        # 11. Clean up
        forecast = forecast.dropna()
        
        # Convert boolean columns to int
        bool_cols = forecast.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            forecast[col] = forecast[col].astype(int)
        
        # Drop EndTime if present
        forecast = forecast.drop('EndTime', axis=1, errors='ignore')
        
        logging.info(f"Aggregation complete. Final shape: {forecast.shape}")
        return forecast
    
    def save(self, df, filename=None, output_dir=None):
        """Save aggregated data to CSV."""
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = "aggregated_data.csv"
        
        path = output_dir / filename
        df.to_csv(path)
        logging.info(f"Saved to {path}")
        return path


def run_aggregation(config_file, instrument, start, end, granularity, output_dir=None, use_ib=False):
    """
    Main entry point for data aggregation.
    
    Args:
        config_file: Path to oanda.cfg
        instrument: Trading pair
        start, end: Date range
        granularity: OANDA granularity
        output_dir: Where to save output
        use_ib: Whether to use IBKR source
        
    Returns:
        Path to saved CSV
    """
    # For IBKR, we might not pass a config file string if we want defaults, 
    # but Aggregator init handles it.
    aggregator = SampoDataAggregator(config_file)
    df = aggregator.run(instrument, start, end, granularity, use_ib=use_ib)
    
    filename = f"{instrument}_{granularity}_output.csv"
    path = aggregator.save(df, filename, output_dir)
    
    return path


def aggregate_and_prepare_dataset(config_file="oanda.cfg", api_key=None, instrument="EUR_USD", 
                                   lookback_days=365, output_dir=None, granularity="D"):
    """
    Wrapper for pipeline compatibility.
    """
    to_time = datetime.utcnow()
    from_time = to_time - pd.Timedelta(days=lookback_days)
    
    start_str = from_time.strftime("%Y-%m-%d")
    end_str = to_time.strftime("%Y-%m-%d")
    
    return run_aggregation(
        config_file=config_file,
        instrument=instrument,
        start=start_str,
        end=end_str,
        granularity=granularity,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sampo Data Aggregation")
    parser.add_argument("--config", type=str, default="oanda.cfg", help="OANDA config file")
    parser.add_argument("--instrument", type=str, default="EUR_USD", help="Trading pair")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2019-12-31", help="End date")
    parser.add_argument("--granularity", type=str, default="D", help="Granularity (D, H1, M15)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--use-ib", action="store_true", help="Use Interactive Brokers")
    
    args = parser.parse_args()
    
    path = run_aggregation(
        config_file=args.config,
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        granularity=args.granularity,
        output_dir=args.output,
        use_ib=args.use_ib
    )
    print(f"Data saved to: {path}")
