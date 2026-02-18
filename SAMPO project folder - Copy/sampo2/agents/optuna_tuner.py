
import optuna
import json
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sampo2.env.trading_env import TradingEnv
from sampo2.config import COMBINED_DATA_FILE, OUTPUT_DIR
import json
import time

class TuningProgressCallback(BaseCallback):
    def __init__(self, trial_number, total_timesteps, verbose=0):
        super(TuningProgressCallback, self).__init__(verbose)
        self.trial_number = trial_number
        self.total_timesteps = total_timesteps
        self.status_file = OUTPUT_DIR / "tuning_status.json"
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Update every 1000 steps
            status = {
                "trial": self.trial_number,
                "current_step": self.num_timesteps,
                "total_steps": self.total_timesteps,
                "progress": self.num_timesteps / self.total_timesteps,
                "timestamp": time.time()
            }
            with open(self.status_file, 'w') as f:
                json.dump(status, f)
        return True


def load_data():
    import numpy as np
    
    # Attempt to load the final dataset used by trainer
    final_path = OUTPUT_DIR / "final_combined_dataset.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            df.sort_index(inplace=True)
        
        # Clean data (mirroring trainer.py)
        df = df.select_dtypes(include=[np.number, bool])
        for col in df.select_dtypes(include=['bool']).columns:
            df[col] = df[col].astype(int)
            
        return df
    else:
        # Fallback (should not happen if pipeline run correctly)
        from sampo2.data.loader import DataLoader
        loader = DataLoader(COMBINED_DATA_FILE)
        return loader.load_data()

def optimize_ppo(trial, trial_number=0):
    df = load_data()

    # Split Data (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # --- Suggest Hyperparameters ---
    
    # 1. Learning Rate (Log scale usually best)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    
    # 2. Entropy Coefficient (Crucial for exploration vs exploitation)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    
    # 3. Clip Range
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # 4. Gamma
    gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
    
    # 5. GAE Lambda
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 0.99, 1.0])
    
    # 6. Batch Size & N_Steps
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    
    # 7. Risk Framework Parameters
    # Removing 'profit_bonus_mult' and old discrete shapers.
    # Searching continuous risk params.
    
    vol_target = trial.suggest_float("vol_target", 0.005, 0.02)
    dd_limit = trial.suggest_float("dd_limit", 0.02, 0.10)
    streak_kill = trial.suggest_int("streak_kill", 3, 10)
    
    lambda_tc = trial.suggest_float("lambda_tc", 0.0, 0.2)
    lambda_vol = trial.suggest_float("lambda_vol", 0.1, 2.0)
    lambda_dd = trial.suggest_float("lambda_dd", 0.5, 5.0)
    lambda_streak = trial.suggest_float("lambda_streak", 0.1, 2.0)
    
    if batch_size > n_steps:
        batch_size = n_steps # Safety fix
        
    # Create Env with Tunable Params
    env_kwargs = {
        'vol_target': vol_target,
        'dd_limit': dd_limit,
        'streak_kill': streak_kill,
        'lambda_tc': lambda_tc,
        'lambda_vol': lambda_vol,
        'lambda_dd': lambda_dd,
        'lambda_streak': lambda_streak,
        # Legacy args if needed but we removed them from TradingEnv __init__
    }
    
    env = DummyVecEnv([lambda: TradingEnv(train_df, **env_kwargs)])
    
    model = PPO("MlpPolicy", 
                env, 
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                clip_range=clip_range,
                batch_size=batch_size,
                n_steps=n_steps,
                verbose=0)
    
    # Train
    from sampo2.utils.progress import SB3ProgressCallback
    progress_callback = SB3ProgressCallback(20000, description=f"Trial {trial_number}")
    callback = TuningProgressCallback(trial_number, 20000)
    
    # Combine callbacks
    model.learn(total_timesteps=20000, callback=[callback, progress_callback])

    # Evaluate on Validation Set
    eval_env = TradingEnv(val_df, **env_kwargs)

    obs, _ = eval_env.reset()
    done = False
    
    # Track Equity Curve
    equity_curve = [eval_env.equity]
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        equity_curve.append(info['equity'])
        
    # Calculate Sharpe Implementation
    from sampo2.agents.risk_evaluator import RiskEvaluator
    
    evaluator = RiskEvaluator(equity_curve)
    metrics = evaluator.calculate_metrics()
    
    sharpe = metrics['sharpe']
    final_equity = info['equity']
    
    print(f"Trial {trial_number}: Sharpe={sharpe:.2f}, Equity={final_equity:.2f}")
    
    # Optimization Objective: Sharpe Ratio
    return sharpe

def run_tuning(n_trials=5):
    print("Starting Optuna Hyperparameter Optimization...")
    print("Objective: Maximize Validation Sharpe Ratio")
    
    study = optuna.create_study(direction="maximize")
    
    # Run optimization
    study.optimize(lambda trial: optimize_ppo(trial, trial.number), n_trials=n_trials) 

    
    print("\n------------------------------------------------")
    print("Optimization Complete!")
    print("Best params:", study.best_params)
    print("Best sharpe:", study.best_value)
    
    # Save Best Params
    output_path = OUTPUT_DIR / "best_hyperparameters.json"
    with open(output_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"Best hyperparameters saved to {output_path}")

if __name__ == "__main__":
    run_tuning(n_trials=5)
