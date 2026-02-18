
import os
import glob
import sys
import time
import math

# FIX PROTOBUF ERROR
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: Tensorboard not installed in this environment.")
    sys.exit(1)

LOG_DIR = "logs/final_agent/PPO_1"
TOTAL_STEPS = 5_000_000

def get_stats():
    # Find event file
    files = glob.glob(os.path.join(LOG_DIR, "events.out.tfevents*"))
    if not files:
        return 0, None
    
    # Pick most recent
    latest_file = max(files, key=os.path.getctime)
    
    try:
        # Load scalars
        event_acc = EventAccumulator(latest_file)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        
        step = 0
        reward = None
        
        # 1. Get Step
        if 'time/total_timesteps' in tags:
            events = event_acc.Scalars('time/total_timesteps')
            if events: step = int(events[-1].value)
        elif 'train/loss' in tags:
             events = event_acc.Scalars('train/loss')
             if events: step = int(events[-1].step)
             
        # 2. Get Reward
        if 'rollout/ep_rew_mean' in tags:
            events = event_acc.Scalars('rollout/ep_rew_mean')
            if events: reward = float(events[-1].value)
            
        return step, reward
    except Exception as e:
        return 0, None

def draw_progress_bar(current, total, reward, length=30):
    percent = float(current) / total
    percent = min(1.0, percent) # Cap at 100%
    
    arrow = '=' * int(round(percent * length) - 1) + '>'
    spaces = ' ' * (length - len(arrow))
    
    # Color coding reward (Basic string hack if terminal supports it, else plain)
    rew_str = "N/A"
    if reward is not None:
        rew_str = f"{reward:+.2f}"
    
    # \r overwrites the line
    sys.stdout.write(f"\rProgress: [{arrow + spaces}] {int(percent * 100)}% ({current:,} / {total:,}) | Avg Reward: {rew_str}   ")
    sys.stdout.flush()

if __name__ == "__main__":
    print(f"Reading Logs from: {LOG_DIR}")
    print("Tracking Progress & Reward (Ctrl+C to stop)...")
    print("-" * 75)
    
    while True:
        step, reward = get_stats()
        draw_progress_bar(step, TOTAL_STEPS, reward)
        
        if step >= TOTAL_STEPS:
            print("\nDone!")
            break
            
        time.sleep(10) # Refresh every 10s
