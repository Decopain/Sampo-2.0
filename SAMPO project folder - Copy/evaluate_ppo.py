
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sampo2.env.trading_env import TradingEnv

# Config
DATA_PATH = "updated2022-2023_july2july.csv"
MODEL_PATH = "models/ppo_robust_test"
OUTPUT_METRICS_PATH = "evaluation_metrics.txt"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading real data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # Parse Time
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
        # Map Columns to Env expectations
        column_map = {
            'Blue': 'renko_prob_blue',
            'Red': 'renko_prob_red',
            'Red/Yellow': 'renko_prob_ry',
            'Blue/Yellow': 'renko_prob_by',
            'volume': 'Volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        # Fill NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(0.0, inplace=True)
        
        print(f"Loaded Data: {df.shape} rows.")
        return df
    else:
        print("Data file not found. Please ensure updated2022-2023_july2july.csv exists.")
        return None

def evaluate():
    df = load_data()
    if df is None: return

    # create env
    env = DummyVecEnv([lambda: TradingEnv(df)])
    
    # load model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Running evaluation loop...")
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0] # VecEnv returns array
        
        if done[0]:
            break
            
    # Extract Trade Log from Info
    # Note: DummyVecEnv wraps info in a list
    final_info = info[0]
    trades = final_info.get('trades', [])
    equity = final_info.get('equity', 0.0)
    
    # Calculate Metrics
    n_trades = len(trades)
    if n_trades > 0:
        df_trades = pd.DataFrame(trades)
        
        # Metrics
        total_pnl = df_trades['PnL'].sum()
        win_rate = (df_trades['PnL'] > 0).mean()
        avg_trade = df_trades['PnL'].mean()
        
        max_dd = final_info.get('dd', 0.0)
        
        print("\n--- Evaluation Results ---")
        print(f"Total Trades: {n_trades}")
        print(f"Final Equity: {equity:.2f}")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg PnL per Trade: {avg_trade:.2f}")
        print(f"Max Drawdown (End): {max_dd:.2%}")
        
        # Detailed Breakdown
        print("\n--- Trade Analysis ---")
        print(df_trades['Reason'].value_counts())
        
        # Save to file
        with open(OUTPUT_METRICS_PATH, "w") as f:
            f.write(f"Total Trades: {n_trades}\n")
            f.write(f"Final Equity: {equity:.2f}\n")
            f.write(f"Total PnL: {total_pnl:.2f}\n")
            f.write(f"Win Rate: {win_rate:.2%}\n")
            f.write(f"Avg PnL per Trade: {avg_trade:.2f}\n")
            f.write(f"Max Drawdown: {max_dd:.2%}\n")
            f.write("\nExit Reasons:\n")
            f.write(df_trades['Reason'].value_counts().to_string())

    else:
        print("No trades executed.")

if __name__ == "__main__":
    evaluate()
