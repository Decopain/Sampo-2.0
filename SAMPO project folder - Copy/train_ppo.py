
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sampo2.env.trading_env import TradingEnv


# Config
DATA_PATH = "outputs/tuning_dataset_2006_2010.csv" 
TOTAL_TIMESTEPS = 50000
MODEL_PATH = "models/ppo_robust_test"
LOG_DIR = "logs/ppo_robust_test"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading real data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # Parse Time
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
        # Map Columns to Env expectations
        # Env expects: renko_prob_blue, renko_prob_red, renko_prob_ry, renko_prob_by
        # CSV has: Blue, Red, Red/Yellow, Blue/Yellow
        
        column_map = {
            'Blue': 'renko_prob_blue',
            'Red': 'renko_prob_red',
            'Red/Yellow': 'renko_prob_ry',
            'Blue/Yellow': 'renko_prob_by',
            'volume': 'Volume' 
            # Note: Env checks lowercase 'atr' which is already in CSV
        }
        df.rename(columns=column_map, inplace=True)
        
        # Fill NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(0.0, inplace=True)
        
        print(f"Loaded Data: {df.shape} rows.")
        return df
    else:
        print("Real data not found. Generating Dummy Data...")
        dates = pd.date_range(start='2020-01-01', periods=5000, freq='5min')
        import numpy as np
        df = pd.DataFrame({
            'Open': 1.1 + np.random.randn(5000).cumsum()*0.001,
            'High': 1.1 + np.random.randn(5000).cumsum()*0.001 + 0.0005,
            'Low': 1.1 + np.random.randn(5000).cumsum()*0.001 - 0.0005,
            'Close': 1.1 + np.random.randn(5000).cumsum()*0.001,
            'Volume': np.random.randint(100, 1000, 5000),
            'atr': [0.0005]*5000,
            'renko_prob_blue': np.random.rand(5000),
            'renko_prob_red': np.random.rand(5000),
            'renko_prob_ry': np.random.rand(5000), 
            'renko_prob_by': np.random.rand(5000)
        }, index=dates)
        return df

def train():
    # 1. Load Data
    df = load_data()

    # 2. Create Env
    env = DummyVecEnv([lambda: TradingEnv(df)])
    
    # 3. Setup Model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    
    # 4. Train
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # 5. Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train()
