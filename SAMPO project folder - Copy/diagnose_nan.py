
import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO

# Add Project Root
sys.path.append(os.getcwd())
try:
    from sampo2.env.trading_env import TradingEnv
except ImportError:
    print("Error importing TradingEnv. Make sure you run from project root.")
    sys.exit(1)

# Config
DATASET_PATH = "outputs/tuning_dataset_2006_2010.csv"
CHECKPOINT_PATH = "models/final_agent/sampo_trial25_3300000_steps.zip"

ENV_KWARGS = {
    "conf_flip": 0.55,
    "cooldown_steps": 4,
    "profit_buffer": 0.000486,
    "t_profit_steps": 10,
    "max_hold_steps": 278,
    "close_bonus": 0.0,
    "inactivity_penalty": 0.001
}

def diagnose():
    print(f"Loading Data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    df = df.dropna().sort_index()
    
    # Feature Engineering (Same as Resume)
    df['Hour_norm'] = df.index.hour / 23.0
    df['Day_norm'] = df.index.dayofweek / 4.0
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_norm'] = (df['High'] - df['Low']) / df['Close']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    print("Creating Env...")
    env = TradingEnv(df, **ENV_KWARGS)
    
    print(f"Loading Model {CHECKPOINT_PATH}...")
    model = PPO.load(CHECKPOINT_PATH)
    
    print("Starting Step-by-Step Diagnostic...")
    obs, info = env.reset()
    
    # Check Initial Obs
    if np.isnan(obs).any():
        print("❌ FATAL: NaN in Initial Observation!")
        print(f"Obs: {obs}")
        return

    for i in range(2000): # Run for 2000 steps (cover 1024 batch)
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        
        if np.isnan(action).any():
            print(f"❌ FATAL: Model produced NaN Action at step {i}!")
            print(f"Obs Input was: {obs}")
            return
            
        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        # Check Obs
        if np.isnan(obs).any():
            print(f"❌ FATAL: Environment produced NaN Observation at step {i}!")
            print(f"Last Action: {action}")
            print(f"Reward: {reward}")
            print(f"Indices of NaN: {np.where(np.isnan(obs))}")
            # Identify which feature
            # Standard Obs: [Window... , Balance, Pos, DD, Streak, ExecState...]
            # ExecState: [Unrealized, Hold, Cost, DistTP, DistSL, DistTrail, Chop]
            return
            
        # Check Reward
        if np.isnan(reward) or np.isinf(reward):
            print(f"❌ FATAL: Environment produced NaN/Inf Reward at step {i}!")
            return
            
        if done:
            print(f"Episode Done at step {i}. Resetting.")
            obs, info = env.reset()
            
        if i % 100 == 0:
            print(f"Step {i}: OK (Equity: {env.equity:.2f})")
            
    print("✅ Diagnostic Passed: No NaNs found in 2000 steps.")

if __name__ == "__main__":
    diagnose()
