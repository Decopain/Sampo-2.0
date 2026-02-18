
import sys
from sampo2.agents.optuna_tuner import run_tuning
from sampo2.agents.trainer import PPOTrainer

def main():
    print("==================================================")
    print("SAMPO2 Automated Training Pipeline")
    print("Step 1: Hyperparameter Optimization (5 Trials, 20k steps each)")
    print("Step 2: PPO Agent Training (50k steps)")
    print("==================================================\n")
    
    # Step 1: Tuning
    print(">>> Starting Step 1: Tuning...")
    try:
        run_tuning(n_trials=5)
    except Exception as e:
        print(f"Error during tuning: {e}")
        sys.exit(1)
        
    print("\n[Step 1 Complete] Best hyperparameters saved.")
    
    # Step 2: Training
    print("\n>>> Starting Step 2: Training Agent with Optimized Params...")
    try:
        trainer = PPOTrainer()
        # Note: Trainer automatically loads 'best_hyperparameters.json' in __init__ or train
        trainer.train(total_timesteps=50000)
        
        # Evaluate
        print("\n>>> Evaluating Final Model...")
        trainer.evaluate()
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
        
    print("\n==================================================")
    print("[SUCCESS] Pipeline Complete. Agent Ready.")
    print("==================================================")

if __name__ == "__main__":
    main()
