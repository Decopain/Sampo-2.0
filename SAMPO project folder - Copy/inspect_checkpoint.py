
from stable_baselines3 import PPO
import numpy as np
import os

CHECKPOINT_PATH = "models/final_agent/sampo_trial25_3300000_steps.zip"

def inspect():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"File not found: {CHECKPOINT_PATH}")
        return

    print(f"Loading {CHECKPOINT_PATH}...")
    try:
        model = PPO.load(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    params = model.get_parameters()
    
    print("Checking parameters for NaNs...")
    has_nan = False
    for key, value in params['policy'].items():
        if np.isnan(value).any():
            print(f"❌ NaN detected in {key}!")
            has_nan = True
        if np.isinf(value).any():
            print(f"❌ Inf detected in {key}!")
            has_nan = True
            
    # Check log_std specifically
    if 'log_std' in params['policy']:
        log_std = params['policy']['log_std']
        print(f"Log Std Stats: Min={log_std.min()}, Max={log_std.max()}, Mean={log_std.mean()}")
        print(f"Log Std Values: {log_std}")
    else:
        print("Warning: log_std not found in policy params dict keys: ", params['policy'].keys())
            
    if not has_nan:
        print("PASS: No NaNs/Infs found in Policy Weights.")
        
    # Check Optimizer if available (SB3 doesn't easily expose optimizer state via get_parameters on load unless we dig)
    # But checking weights is the most important step.

if __name__ == "__main__":
    inspect()
