import pandas as pd
import numpy as np
from sampo2.env.trading_env import TradingEnv

def debug_env():
    # Load small chunk of data
    df = pd.read_csv("outputs/tuning_dataset_2006_2010.csv", index_col=0, parse_dates=True).iloc[0:1000]
    df = df.select_dtypes(include=[np.number]).fillna(0)
    
    env = TradingEnv(
        df,
        conf_flip=0.51, # Super low threshold
        inactivity_penalty=0.0001,
        streak_kill=100
    )
    
    obs = env.reset()
    print("Initial Balance:", env.balance)
    print("Initial Position:", env.position)
    
    # Run 100 steps with aggressive actions
    for i in range(100):
        # Force a "buy" signal in the action (e.g., 1.0)
        # Action space is Box(-1, 1)
        # Step logic: target_exposure = action[0] (if using continuous?)
        # Wait, check step code: action is usually just passed to logic
        
        action = np.array([0.8], dtype=np.float32) # Strong Buy
        obs, reward, done, trunc, info = env.step(action)
        
        if i < 5:
            print(f"Step {i}: Action={action}, Position={env.position}, Reward={reward:.6f}, Trades={len(env.trade_log)}")
            
    print(f"Final Trades: {len(env.trade_log)}")
    print(f"Final Balance: {env.balance}")
    
if __name__ == "__main__":
    debug_env()
