
import pandas as pd
import sys

def main():
    try:
        df = pd.read_csv("c:/Users/Jeff/Desktop/sample_DS/outputs/evaluation_trades.csv")
        total_trades = len(df)
        winning_trades = df[df['PnL'] > 0]
        losing_trades = df[df['PnL'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = df['PnL'].sum()
        avg_pnl = df['PnL'].mean()
        
        # Simple Drawdown Calculation (cumulative PnL)
        df['Cumulative_PnL'] = df['PnL'].cumsum()
        peak_pnl = df['Cumulative_PnL'].cummax()
        drawdown = df['Cumulative_PnL'] - peak_pnl
        max_drawdown = drawdown.min()

        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL (pips/units): {total_pnl:.5f}")
        print(f"Average PnL per Trade: {avg_pnl:.5f}")
        print(f"Max Drawdown: {max_drawdown:.5f}")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()
