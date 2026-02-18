
import numpy as np
import pandas as pd

class RiskEvaluator:
    def __init__(self, equity_curve, trade_log=None):
        """
        Args:
            equity_curve (list or np.array): Time series of account equity.
            trade_log (list, optional): List of trade dictionaries.
        """
        self.equity = np.array(equity_curve)
        self.trade_log = trade_log if trade_log is not None else []
        self.returns = np.diff(self.equity) / self.equity[:-1]
        self.returns = np.nan_to_num(self.returns) # Handle potential 0 division
        
    def calculate_metrics(self):
        metrics = {}
        
        # Safety Check
        if len(self.equity) < 2:
            return {
                'total_return': 0.0,
                'volatility_ann': 0.0,
                'sharpe': 0.0,
                'sortino': 0.0,
                'max_drawdown': 0.0,
                'calmar': 0.0,
                'win_rate': 0.0,
                'trade_count': 0,
                'turnover': 0,
                'cvar_95': 0.0
            }
            
        # 1. Total Return
        metrics['total_return'] = (self.equity[-1] - self.equity[0]) / self.equity[0]
        
        # 2. Volatility (Annualized)
        # Assuming Daily Steps? If H1, then 252*24?
        # Let's assume steps are "periods" and report per-period stats or annualize generically (252 days)
        # If input data is H1, we should scale by sqrt(252*24). If D, sqrt(252).
        # We'll use a standard crypto/fx assumption: 365*24 = 8760 hours/year or 252 days
        SCALE = np.sqrt(252 * 24) # Assuming H1 steps
        
        std = np.std(self.returns)
        metrics['volatility_ann'] = std * SCALE
        
        # 3. Sharpe
        # (Mean Return / Std Dev) * Sqrt(Periods)
        mean_ret = np.mean(self.returns)
        metrics['sharpe'] = (mean_ret / (std + 1e-9)) * SCALE
        
        # 4. Sortino (Downside Deviation)
        downside_returns = self.returns[self.returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        metrics['sortino'] = (mean_ret / (downside_std + 1e-9)) * SCALE
        
        # 5. Drawdown Stats
        peak = np.maximum.accumulate(self.equity)
        drawdown = (peak - self.equity) / peak
        metrics['max_drawdown'] = np.max(drawdown)
        metrics['avg_drawdown'] = np.mean(drawdown)
        
        # 6. Calmar (Annualized Return / Max DD)
        # Annualized Return approx
        n_periods = len(self.returns)
        years = n_periods / (252*24)
        if years > 0:
            cagr = (self.equity[-1] / self.equity[0]) ** (1/years) - 1
            metrics['calmar'] = cagr / (metrics['max_drawdown'] + 1e-9)
        else:
            metrics['calmar'] = 0
            
        # 7. Win Rate (of steps with non-zero return)
        active_returns = self.returns[np.abs(self.returns) > 1e-6]
        if len(active_returns) > 0:
            metrics['win_rate'] = np.mean(active_returns > 0)
        else:
            metrics['win_rate'] = 0
        # 8. Trade Stats
        metrics['trade_count'] = len(self.trade_log) # Standardized key
        metrics['total_trades'] = len(self.trade_log)
        
        # 9. Turnover (Sum of |size opened + size closed|)
        if self.trade_log:
            try:
                # Assume Round Trip = Entry + Exit. Total Size * 2 roughly.
                # 'Size' in log is the closed amount.
                turnover = sum(t.get('Size', 0) * 2.0 for t in self.trade_log)
                metrics['turnover'] = turnover
            except:
                metrics['turnover'] = 0
        else:
            metrics['turnover'] = 0
            
        # 10. CVaR(95)
        if len(self.returns) > 0:
            var_95 = np.percentile(self.returns, 5)
            cvar_95 = self.returns[self.returns <= var_95].mean()
            metrics['cvar_95'] = cvar_95
        else:
            metrics['cvar_95'] = 0
            
        return metrics

    def get_report(self):
        m = self.calculate_metrics()
        report = f"""
        Risk Analysis Report
        --------------------
        Total Return: {m['total_return']:.2%}
        Max Drawdown: {m['max_drawdown']:.2%}
        Sharpe Ratio: {m['sharpe']:.2f}
        Sortino Ratio: {m['sortino']:.2f}
        Calmar Ratio: {m['calmar']:.2f}
        Volatility:   {m['volatility_ann']:.2%}
        Win Rate:     {m['win_rate']:.2%}
        Total Trades: {m['total_trades']}
        """
        return report
