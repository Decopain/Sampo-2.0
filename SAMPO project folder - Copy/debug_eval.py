
import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Add project path
sys.path.append(os.getcwd())
from sampo2.env.trading_env import TradingEnv

def debug_eval():
    print("Loading Model...")
    model_path = "best_tuned_model.zip"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Using dummy model? No, aborting.")
        return

    model = PPO.load(model_path)
    
    print("Loading Data...")
    df = pd.read_csv("outputs/tuning_dataset_2006_2010.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].fillna(0)
    
    # Use Training Split logic from tuner
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    
    print(f"Data Loaded. Rows: {len(train_df)}")
    
    env = TradingEnv(train_df, obs_cols=['log_ret', 'vol_norm'])
    
    obs, info = env.reset()
    done = False
    
    steps = 0
    trades = 0
    actions = []
    
    print("Starting Loop...")
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        steps += 1
        actions.append(action[0])
        
        if steps % 1000 == 0:
            print(f"Step {steps}: Action {action[0]:.4f} | pos {env.position:.2f}")

        if done:
            print(f"DONE at Step {steps}")
            print(f"Reason: {info.get('terminal_observation', 'Unknown')}") # Actually info usually has terminal info
            print(f"Logs in info: {len(info.get('trades', []))}")
            break
            
    print(f"Total Steps: {steps}")
    print(f"Avg Action: {np.mean(actions):.4f}")
    print(f"Action Std: {np.std(actions):.4f}")

if __name__ == "__main__":
    debug_eval()
