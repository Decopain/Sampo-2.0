import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sampo2.env.trading_env import TradingEnv

# Config
# Using the verified data file found in the project root
DATA_PATH = "updated2022-2023_july2july.csv"
# Reduced timesteps for smoke test
TOTAL_TIMESTEPS = 1000
MODEL_PATH = "models/ppo_smoke_test"
LOG_DIR = "logs/ppo_smoke_test"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading real data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # Parse Time
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
        # Map Columns to Env expectations
        # Based on train_ppo.py logic and verified CSV headers
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
        raise FileNotFoundError(f"Critical Data File Missing: {DATA_PATH}")

def run_smoke_test():
    print("--- Starting SAMPO2 Smoke Test ---")
    
    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"FAILED to load data: {e}")
        return

    # 2. Create Env
    print("Initializing Environment...")
    try:
        # Check for 'Close' column which is critical
        if 'Close' not in df.columns:
            print("ERROR: 'Close' column missing from dataframe.")
            return
            
        env = DummyVecEnv([lambda: TradingEnv(df)])
    except Exception as e:
        print(f"FAILED to initialize TradingEnv: {e}")
        return
    
    # 3. Setup Model
    print("Initializing PPO Model...")
    try:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    except Exception as e:
        print(f"FAILED to create PPO model: {e}")
        return
    
    # 4. Train
    print(f"Running short training loop ({TOTAL_TIMESTEPS} steps)...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"FAILED during model.learn(): {e}")
        return
    
    # 5. Save
    try:
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"FAILED to save model: {e}")
        return

    print("--- Smoke Test PASSED ---")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    run_smoke_test()
