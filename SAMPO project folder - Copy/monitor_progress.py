import os
import time
from pathlib import Path
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Constants
TENSORBOARD_DIR = Path("outputs/tensorboard")
TARGET_STEPS = 1_000_000
N_STEPS = 1024  # From our best_hyperparameters.json

def get_latest_run_dir(base_dir):
    """Finds the most recently modified PPO_x directory."""
    if not base_dir.exists():
        return None
    
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("PPO_")]
    if not subdirs:
        return None
    
    def extract_id(name):
        try:
            return int(name.split('_')[-1])
        except ValueError:
            return -1
            
    subdirs.sort(key=lambda x: extract_id(x.name))
    return subdirs[-1]

def monitor():
    run_dir = get_latest_run_dir(TENSORBOARD_DIR)
    if not run_dir:
        print(f"No training runs found in {TENSORBOARD_DIR}")
        return

    print(f"Monitoring Training Run: {run_dir.name}")
    print(f"Target Steps: {TARGET_STEPS:,}")
    print(f"Steps per iteration: {N_STEPS}")
    print("Loading initial data...")
    
    # Initialize Progress Bar
    pbar = tqdm(total=TARGET_STEPS, unit="step", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    last_step = 0
    
    while last_step < TARGET_STEPS:
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            
            # Use 'time/fps' to count iterations (each logged event = 1 iteration)
            if 'time/fps' in ea.Tags().get('scalars', []):
                events = ea.Scalars('time/fps')
                if events:
                    # Number of events = number of iterations completed
                    iterations = len(events)
                    current_step = iterations * N_STEPS
                    
                    if current_step > last_step:
                        update_amount = current_step - last_step
                        pbar.update(update_amount)
                        last_step = current_step
        except Exception as e:
            pass  # Silently retry
            
        if last_step >= TARGET_STEPS:
            break
            
        time.sleep(5)  # Check every 5 seconds
        
    pbar.close()
    print("\nTraining Complete!")

if __name__ == "__main__":
    monitor()
