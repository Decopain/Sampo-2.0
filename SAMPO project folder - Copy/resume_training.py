
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from scheduler import linear_schedule # Import Helper

import sys
sys.path.append(os.getcwd())
from sampo2.env.trading_env import TradingEnv

# --- CONFIG ---
CHECKPOINT_PATH = "models/final_agent/sampo_trial25_3300000_steps.zip"
DATASET_PATH = "outputs/tuning_dataset_2006_2010.csv"
TOTAL_TIMESTEPS = 1_700_000  # Remaining steps (5M - 3.3M)
CHECKPOINT_FREQ = 100_000
MODEL_DIR = "models/final_agent"
LOG_DIR = "logs/final_agent"

# Params from Trial 25 (Needed for Env Init)
ENV_KWARGS = {
    "conf_flip": 0.55,
    "cooldown_steps": 4,
    "profit_buffer": 0.000486,
    "t_profit_steps": 10,
    "max_hold_steps": 278,
    "close_bonus": 0.0,
    "inactivity_penalty": 0.001,
    "streak_kill": 100, # Relaxed for training resume
    "dd_kill": 0.25     # Relaxed for training resume
}

class StatusCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.status_file = "current_training_status.json"
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            try:
                inner_env = self.training_env.envs[0]
                status = {
                    "step": self.num_timesteps + 3300000, # Offset
                    "total_steps": 5000000,
                    "progress": (self.num_timesteps + 3300000) / 5000000,
                    "equity": float(inner_env.equity),
                    "trade_count": len(inner_env.trade_log),
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S")
                }
                with open(self.status_file, "w") as f:
                    json.dump(status, f)
            except:
                pass
        return True

class EntropyAnnealingCallback(BaseCallback):
    """
    Linearly decays entropy coefficient from initial to final value.
    Note: SB3's ent_coef does not support schedules natively.
    """
    def __init__(self, initial_val: float, final_val: float, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.initial_val = initial_val
        self.final_val = final_val
        self.total_steps = total_steps
        
    def _on_step(self) -> bool:
        # Calculate progress (0.0 to 1.0)
        # self.num_timesteps counts from 0 in this new run
        progress = min(1.0, self.num_timesteps / self.total_steps)
        current_ent = self.initial_val + (self.final_val - self.initial_val) * progress
        self.model.ent_coef = current_ent
        
        # Log occasionally
        if self.n_calls % 1000 == 0:
            pass # We could log to tensorboard here if needed
            
        return True

def resume():
    # ... (Data loading same) ...
    
    print(f"Loading Data from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    df = df.dropna().sort_index()
    
    # Feature Engineering
    df['Hour_norm'] = df.index.hour / 23.0
    df['Day_norm'] = df.index.dayofweek / 4.0
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_norm'] = (df['High'] - df['Low']) / df['Close']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Create Env
    env = TradingEnv(df, **ENV_KWARGS)
    
    # --- SAFER RESUME STRATEGY (V6: RESET EXTREME SIGMA) ---
    # Diagnosis: The checkpoint has log_std=43 (Sigma=5e18). This causes NaN/Explosions.
    # Fix: Manually reset log_std to 0.0 (Sigma=1.0) or -1.0 (Sigma=0.36).
    # We choose -0.5 (Sigma=0.6) as a reasonable starting point for exploration.
    
    print("Initializing FRESH Model with CORRECTED SIGMA...")
    
    # Standard LR (back to scheduled)
    lr_schedule = linear_schedule(3e-4, 1e-5) 
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=1024,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01, # Low entropy to discourage explosion again
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cpu"
    )
    
    print(f"Injecting Weights from: {CHECKPOINT_PATH}...")
    model.set_parameters(CHECKPOINT_PATH)
    
    print("SURGICAL INTERVENTION: Resetting Policy Log_Std from ~43 to -0.5...")
    import torch
    with torch.no_grad():
        # Fill log_std with -0.5 (approx std=0.6)
        # Note: SB3 implementation might have log_std as a parameter vector
        model.policy.log_std.fill_(-0.5)
        
    print(f"Resuming Training for {TOTAL_TIMESTEPS} steps (Target: 5M)...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sampo_trial25_resumed"
    )
    
    status_callback = StatusCallback()
    
    # Anneal Entropy from 0.01 down to 0.0 over the remaining steps
    entropy_callback = EntropyAnnealingCallback(
        initial_val=0.01,
        final_val=0.0,
        total_steps=TOTAL_TIMESTEPS
    )
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback, status_callback, entropy_callback])
        
        model.save(f"{MODEL_DIR}/sampo_final_5M_completed")
        print("Marathon Complete!")
    except KeyboardInterrupt:
        model.save(f"{MODEL_DIR}/sampo_resumed_interrupted")
        print("Saved Interrupted State.")
    except KeyboardInterrupt:
        model.save(f"{MODEL_DIR}/sampo_resumed_interrupted")
        print("Saved Interrupted State.")

if __name__ == "__main__":
    resume()
