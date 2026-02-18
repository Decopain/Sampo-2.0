
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.append(os.getcwd())
from sampo2.env.trading_env import TradingEnv
from sampo2.agents.risk_evaluator import RiskEvaluator

# --- CONFIG (Trial 25 Golden Set) ---
DATASET_PATH = "outputs/tuning_dataset_2006_2010.csv"
TOTAL_TIMESTEPS = 5_000_000  # 5 Million Steps
CHECKPOINT_FREQ = 100_000
MODEL_DIR = "models/final_agent"
LOG_DIR = "logs/final_agent"

# Params from Trial 25
LR = 0.00030957 # 3.1e-4
BATCH_SIZE = 128
CLIP_RANGE = 0.102
ENT_COEF = 0.09476
N_STEPS = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95

# Env Params (Trial 25)
ENV_KWARGS = {
    "conf_flip": 0.55,
    "cooldown_steps": 4,
    "profit_buffer": 0.000486,
    "t_profit_steps": 10,
    "max_hold_steps": 278,
    "close_bonus": 0.0,
    "inactivity_penalty": 0.001
}

# Ensure dirs exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class StatusCallback(BaseCallback):
    """Updates dashboard status file during long training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.status_file = "current_training_status.json"
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            # Capture last trades from env
            try:
                # VecEnv wrapper makes accessing env tricky: env.envs[0]
                inner_env = self.training_env.envs[0]
                trades = inner_env.trade_log
                equity = inner_env.equity
                
                status = {
                    "step": self.num_timesteps,
                    "total_steps": TOTAL_TIMESTEPS,
                    "progress": self.num_timesteps / TOTAL_TIMESTEPS,
                    "equity": float(equity),
                    "trade_count": len(trades),
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                
                with open(self.status_file, "w") as f:
                    json.dump(status, f)
            except Exception as e:
                pass # Don't crash training for status update
        return True

def train():
    print(f"Loading Data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    df = df.dropna().sort_index()
    
    # Feature Engineering (Same as Tuner)
    df['Hour_norm'] = df.index.hour / 23.0
    df['Day_norm'] = df.index.dayofweek / 4.0
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_norm'] = (df['High'] - df['Low']) / df['Close']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Data Loaded: {len(df)} rows.")
    
    # Create Env
    env = TradingEnv(df, **ENV_KWARGS)
    
    # Initialize Agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu" # Use CPU to avoid CUDA overhead for simple scalar inputs if GPU not avail
    )
    
    print("Starting Long Training (5M Steps)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sampo_trial25"
    )
    
    status_callback = StatusCallback()
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, status_callback])
        model.save(f"{MODEL_DIR}/sampo_final_5M")
        print("Training Complete. Model Saved.")
    except KeyboardInterrupt:
        print("Training Interrupted by User. Saving Emergency Checkpoint...")
        model.save(f"{MODEL_DIR}/sampo_interrupted")

if __name__ == "__main__":
    train()
