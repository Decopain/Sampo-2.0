
import os
# FIX PROTOBUF CONFLICT (Streamlit vs Tensorboard)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import optuna
import json
import time
import pandas as pd
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from pathlib import Path
from datetime import datetime

# Import project modules
import sys
import os
sys.path.append(os.getcwd())
from sampo2.env.trading_env import TradingEnv
from sampo2.env.actions import ACTION_MAP
from sampo2.agents.risk_evaluator import RiskEvaluator

# CONFIG
LOG_FILE = "tuning_log.json"
STATUS_FILE = "current_trial.json"
RESULTS_FILE = "tuning_results_final.json"
# --- CONFIG ---
DATASET_PATH = "outputs/tuning_dataset_2006_2010.csv"
N_TRIALS = 50       # Focused Batch
N_STEPS = 10000     # Increased fidelity for Fine-Tuning
N_EVAL_EPISODES = 5

CONTROL_FILE = "tuning_control.json"
RESULTS_FILE = "tuning_results_final.json"
STATUS_FILE = "current_trial.json"

def check_control_signals():
    """Poll control file for Pause/Stop commands."""
    if not Path(CONTROL_FILE).exists():
        return # Default to RUNNING
    
    try:
        with open(CONTROL_FILE, 'r') as f:
            cmd = json.load(f).get('command', 'run')
            
        if cmd == 'stop':
            print("STOP Signal Received. Exiting...")
            raise KeyboardInterrupt
            
        elif cmd == 'pause':
            print("PAUSED...", end='\r')
            while cmd == 'pause':
                time.sleep(1)
                # Re-check control file
                try:
                    with open(CONTROL_FILE, 'r') as f:
                        cmd = json.load(f).get('command', 'run')
                except:
                    pass
            print("RESUMED!    ")
            
    except Exception as e:
        print(f"Control Read Error: {e}")

def get_linear_schedule(start_value, end_value):
    """
    Returns a callable schedule function for Stable Baselines3.
    progress_remaining goes from 1.0 (start) to 0.0 (end).
    """
    def func(progress_remaining: float) -> float:
        return end_value + (start_value - end_value) * progress_remaining
    return func

def set_random_seed(seed: int):
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_log(trial_results):
    """Append trial result to log file."""
    try:
        if Path(LOG_FILE).exists():
            with open(LOG_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add timestamp
        trial_results['completed_at'] = datetime.now().isoformat()
        data.append(trial_results)
        
        with open(LOG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Logging Error: {e}")

def update_status(trial_number, step, total_steps, best_sharpe, current_metrics=None, start_time=None):
    """Update the live status file for the dashboard."""
    try:
        status = {
            "trial_number": trial_number,
            "total_trials": N_TRIALS,
            "current_step": step,
            "total_steps": total_steps,
            "progress": min(step / total_steps, 1.0),
            "best_sharpe": best_sharpe,
            "eta": "Calculating...", # TODO: Add real ETA logic
            "last_updated": datetime.now().isoformat()
        }
        if start_time:
            status["start_time"] = start_time
        if current_metrics:
            status["current_metrics"] = current_metrics
            
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f)
    except Exception as e:
        pass # Silent fail on status update to not break training

class TuningObjective:
    def __init__(self, df):
        self.df = df
        
        # Filter Observation Columns (Remove Raw Prices, keep scaled features)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Time_Link']
        self.obs_cols = [c for c in df.columns if c not in exclude_cols]
        
        self.best_sharpe = -np.inf
        self.trial_counter = 0
        self.start_time = time.time() # Track overall start

    def __call__(self, trial):
        self.trial_counter += 1
        start_time = time.time()
        
        # 1. Sample Hyperparameters & Seed
        trial_seed = 42 + self.trial_counter # Deterministic seed per trial sequence
        set_random_seed(trial_seed)
        
        # Execution (Hard Physics)
        conf_flip = trial.suggest_float("conf_flip", 0.51, 0.65, step=0.02) # Lowered to encourage entry
        cooldown_steps = trial.suggest_int("cooldown_steps", 3, 10)
        profit_buffer = trial.suggest_float("profit_buffer", 0.0002, 0.0010)
        t_profit_steps = trial.suggest_int("t_profit_steps", 10, 60)
        max_hold_steps = trial.suggest_int("max_hold_steps", 50, 300)
        
        # PPO (Soft Brain)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef_start = trial.suggest_float("ent_coef", 0.05, 0.10) # BOOSTED ENTROPY (Force Exploration)
        ent_coef_end = 0.001 # Decay to low
        
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048])
        
        # 2. Setup Environment
        env_config = {
            "df": self.df,
            "conf_flip": conf_flip,
            "cooldown_steps": cooldown_steps,
            "profit_buffer": profit_buffer,
            "t_profit_steps": t_profit_steps,
            "max_hold_steps": max_hold_steps,
            "k_tp": 2.0, "k_sl": 1.0, "k_trl": 1.5,
            # Reverted streak_kill to default (strict)
            "inactivity_penalty": 0.001, # 10x penalty to force action
            "obs_cols": self.obs_cols
        }
        
        # Pass seed to env
        def make_env():
            e = TradingEnv(**env_config)
            e.reset(seed=trial_seed) # Ensure env is seeded
            return e
            
        env = DummyVecEnv([make_env])
        # VecNormalize removed (User request: Use Log Returns)
        
        # 3. Setup Model (CPU)
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
        # Schedules
        # LR accepts schedule, ent_coef does NOT in some SB3 versions (needs float/tensor)
        lr_schedule = get_linear_schedule(learning_rate, learning_rate * 0.1)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            clip_range=clip_range,
            ent_coef=ent_coef_start, # Start with float
            batch_size=batch_size,
            n_steps=n_steps,
            device='cpu',
            verbose=0,
            seed=trial_seed, # Seed PPO
            policy_kwargs=policy_kwargs
        )
        
        # 4. Training Loop with Progress Reporting
        # We manually step learn() to update status
        chunk_size = 10000
        for i in range(0, N_STEPS, chunk_size):
            # --- CONTROL SIGNAL CHECK ---
            check_control_signals()
            # ----------------------------
            
            # Manual Entropy Annealing
            progress = i / N_STEPS
            current_ent = ent_coef_start + progress * (ent_coef_end - ent_coef_start)
            model.ent_coef = current_ent
            
            model.learn(total_timesteps=chunk_size, reset_num_timesteps=False)
            
            # --- LIVE METRICS ---
            try:
                # Access the inner environment (DummyVecEnv -> TradingEnv)
                inner_env = env.envs[0]
                
                # We need at least some history
                if hasattr(inner_env, 'equity_history') and len(inner_env.equity_history) > 2:
                    ev = RiskEvaluator(inner_env.equity_history, inner_env.trade_log)
                    m = ev.calculate_metrics()
                    cur_metrics = {
                        "return": m.get('total_return', 0),
                        "sharpe": m.get('sharpe', 0),
                        "max_dd": m.get('max_drawdown', 0),
                        "trades": m.get('trade_count', 0),
                        "win_rate": m.get('win_rate', 0),
                        "turnover": m.get('turnover', 0)
                    }
                else:
                    cur_metrics = {}
            except Exception as e:
                # print(f"Metric Error: {e}")
                cur_metrics = {}

            update_status(self.trial_counter, i+chunk_size, N_STEPS, self.best_sharpe, 
                         current_metrics=cur_metrics, start_time=self.start_time)
            
            # Pruning check (optional - simplified here)
            # if trial.should_prune(): raise optuna.TrialPruned()

        # 5. Evaluation
        # Run ep for metrics
        # (Ideally we'd use a separate validation set, but using train end for speed check in V1)
        obs = env.reset()
        done = [False]
        logs = []
        
        while not done[0]:
            # Use stochastic prediction (False) because we want to see if the exploration found anything
            # Deterministic often freezes in early training stages
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, done, info = env.step(action)
            
            if done[0]:
                # Capture logs from the JUST finished episode
                # info[0] contains the final state info before reset
                # TradingEnv key is 'trades', not 'trade_log'
                logs = info[0].get('trades', [])
                equity_history = info[0].get('equity_history', [])
                break
        
        # 6. Evaluation Logic
        # (Removed old inner_env access which was bugged)
        
        evaluator = RiskEvaluator(equity_history, logs)
        metrics = evaluator.calculate_metrics()
        
        duration = time.time() - start_time
        
        # Update Best
        if metrics['sharpe'] > self.best_sharpe:
            self.best_sharpe = metrics['sharpe']
            model.save("best_tuned_model")
            print(f"New Best Model Saved! Sharpe: {self.best_sharpe:.2f}")
            
        # 6. Log Result
        result_entry = {
            "Trial ID": trial.number,
            "Sharpe": metrics['sharpe'],
            "Return": metrics['total_return'],
            "MaxDD": metrics['max_drawdown'],
            "Sortino": metrics['sortino'],
            "Calmar": metrics['calmar'],
            "Win Rate": metrics['win_rate'],
            "Trade Count": metrics['trade_count'],
            "Turnover": metrics.get('turnover', 0), # Ensure env calculates this
            "Duration": duration,
            "Params": trial.params,
            # Add breakdown if available in env logs
            "conf_flip": conf_flip,
            "cooldown_steps": cooldown_steps
        }
        save_log(result_entry)
        
        # Final Status Update
        update_status(self.trial_counter, N_STEPS, N_STEPS, self.best_sharpe, 
                      current_metrics={"return": metrics['total_return'], "sharpe": metrics['sharpe'], 
                                       "max_dd": metrics['max_drawdown'], "trades": metrics['trade_count']})
        
        return metrics['sharpe']

def main():
    # CLEANUP: Remove old logs for fresh dashboard start
    if Path(STATUS_FILE).exists(): os.remove(STATUS_FILE)
    if Path(LOG_FILE).exists(): os.remove(LOG_FILE)
    
    # Initial Status to wake up dashboard
    update_status(0, 0, N_STEPS, 0.0)
    
    print("Loading Data...")
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    df = df.dropna().sort_index()
    
    # Add Numeric Time Features (so Agent can 'see' time)
    df['Hour_norm'] = df.index.hour / 23.0
    df['Day_norm'] = df.index.dayofweek / 4.0
    
    # Feature Engineering (Stationarity for PPO)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol_norm'] = (df['High'] - df['Low']) / df['Close']
    
    # Drop non-numeric columns and NaN rows
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]
    df = df.replace([np.inf, -np.inf], 0).fillna(0) # Safety fill
    
    # Split Train/Val (Simple split for CPU speed)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    
    study = optuna.create_study(direction="maximize")
    objective = TuningObjective(train_df)
    
    print(f"Starting Tuning with {N_TRIALS} trials (CPU mode)...")
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")
        
    print("Tuning Complete.")
    print(f"Best Params: {study.best_params}")
    
    # Save Best
    with open(RESULTS_FILE, 'w') as f:
        json.dump(study.best_params, f, indent=4)

if __name__ == "__main__":
    main()
