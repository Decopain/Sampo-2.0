import pandas as pd
import numpy as np
from arch import arch_model
from sampo2.config import GARCH_P, GARCH_Q, GARCH_MEAN, RESCALE_FACTOR

class GarchVolatilityModel:
    def __init__(self, p=GARCH_P, q=GARCH_Q, mean=GARCH_MEAN):
        self.p = p
        self.q = q
        self.mean = mean

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """
        Fits GARCH(p,q) and returns conditional volatility.
        Expects returns series (percentage or log returns).
        """
        # Remove NaNs
        clean_returns = returns.dropna()
        if len(clean_returns) < 100:
            print("Warning: Not enough data for GARCH fit.")
            return pd.Series(index=returns.index, data=0.0)

        # Rescale for better convergence (common practice with arch package)
        rescaled_returns = clean_returns * RESCALE_FACTOR

        try:
            model = arch_model(rescaled_returns, vol='Garch', p=self.p, q=self.q, mean=self.mean)
            # fit(disp='off') to suppress output
            res = model.fit(disp='off')
            
            # Get conditional volatility and reverse scaling
            cond_vol = res.conditional_volatility / RESCALE_FACTOR
            
            # Realign with original index
            vol_series = pd.Series(cond_vol, index=clean_returns.index)
            
            # Fill missing (e.g. first row) with 0 or bfill
            aligned_vol = vol_series.reindex(returns.index).fillna(0.0)
            
            return aligned_vol

        except Exception as e:
            print(f"GARCH Fit Error: {e}")
            return pd.Series(index=returns.index, data=0.0)

    def process_timeframes(self, data_loader, df_base, timeframes=['1D', '1H']):
        """
        Generates volatility for multiple timeframes.
        
        Args:
            data_loader: Instance of DataLoader (to use resample)
            df_base: Base dataframe (e.g. 15min or 1H)
            timeframes: List of timeframes to generate
            
        Returns:
            Dictionary {timeframe: volatility_series}
        """
        results = {}
        for tf in timeframes:
            # Resample
            df_resampled = data_loader.resample_data(df_base, tf)
            
            # Calc Returns
            df_resampled = data_loader.add_returns(df_resampled)
            
            # Fit GARCH
            print(f"Fitting GARCH for {tf}...")
            vol = self.fit_predict(df_resampled['Returns'])
            
            results[tf] = vol
            
        return results
