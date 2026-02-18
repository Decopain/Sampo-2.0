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

class PPOTrainer:
    def __init__(self, data_path=COMBINED_DATA_FILE, output_dir=OUTPUT_DIR):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.points_dir = self.output_dir / "checkpoints"
        self.points_dir.mkdir(exist_ok=True)

    def load_data(self):
        # We assume the data is already preprocessed (or we use loader)
        # Ideally, we load the FINAL combined dataset
        combined_path = self.output_dir / "final_combined_dataset.csv"
        if combined_path.exists():
            print(f"Loading Combined Data from {combined_path}")
            df = pd.read_csv(combined_path)
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
                df.sort_index(inplace=True)
            
            # Clean data for Gym Environment
            # Drop non-numeric columns like EndTime_fr
            df = df.select_dtypes(include=[np.number, bool])
            
            # Convert bool to int
            bool_cols = df.select_dtypes(include=['bool']).columns
            for col in bool_cols:
                df[col] = df[col].astype(int)
                
            return df
        else:
            print(f"Warning: Final dataset not found. Falling back to base {self.data_path}")
            loader = DataLoader(self.data_path)
            return loader.load_data()

    def train(self, total_timesteps=10000, resume_path=None, model_name="ppo_final_model", seed=42):
        print(f"--- Starting PPO Training (Model: {model_name}) ---")
        set_random_seed(seed)
        df = self.load_data()
        
        # Split Train/Test?
        # Simple split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"Train Size: {len(train_df)}. Test Size: {len(test_df)}")
        
        # Check for tuned hyperparameters
        hp_path = self.output_dir / "best_hyperparameters.json"
        hyperparams = {}
        env_kwargs = {}
        
        # Default base params
        base_params = {
            'learning_rate': 0.0003,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'batch_size': 64,
            'n_steps': 1024
        }
        
        if hp_path.exists():
            import json
            print(f"Loading optimized hyperparameters from {hp_path}...")
            with open(hp_path, 'r') as f:
                loaded_params = json.load(f)
                
                # Separate Env vs Agent params
                env_keys = ['profit_bonus_mult', 'loss_penalty_mult', 'inactivity_penalty', 
                            'vol_target', 'dd_limit', 'dd_kill', 'streak_kill',
                            'lambda_tc', 'lambda_vol', 'lambda_dd', 'lambda_streak',
                            'conf_flip', 'max_hold_steps', 'cooldown_steps', 'profit_buffer', 't_profit_steps']
                
                # Update env_kwargs
                for k in env_keys:
                    if k in loaded_params:
                        env_kwargs[k] = loaded_params[k]
                
                # Update hyperparams (Agent)
                for k, v in loaded_params.items():
                    if k not in env_keys:
                        hyperparams[k] = v
        else:
            print("No optimized parameters found. Using default settings.")
            hyperparams = base_params.copy()
            # Relax constraints for default run if not tuned
            if 'streak_kill' not in env_kwargs:
                env_kwargs['streak_kill'] = 10 # Relaxed default
        
        # --- Apply Schedules ---
        
        # Learning Rate Schedule (linear decay to 10%)
        lr_start = hyperparams.pop('learning_rate', 3e-4) # default if missing
        hyperparams['learning_rate'] = get_linear_schedule(lr_start, lr_start * 0.1)
        
        # Entropy Schedule (linear decay to 0.001)
        # Handle 'ent_coef' (legacy) or 'ent_coef_start' (tuned)
        if 'ent_coef_start' in hyperparams:
            ent_start = hyperparams.pop('ent_coef_start')
        else:
            ent_start = hyperparams.pop('ent_coef', 0.01)
            
        hyperparams['ent_coef'] = get_linear_schedule(ent_start, 0.001)

        # Initialize Env
        env = DummyVecEnv([lambda: TradingEnv(train_df, **env_kwargs)])

        # Initialize PPO with kwargs
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(self.output_dir / "tensorboard"), seed=seed, **hyperparams)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=str(self.points_dir),
                                                 name_prefix=model_name)
        
        from sampo2.utils.progress import SB3ProgressCallback
        progress_callback = SB3ProgressCallback(total_timesteps, description="PPO Training")
        
        # Train
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, progress_callback])
        
        # Save Final
        final_path = self.output_dir / model_name
        model.save(final_path)
        print(f"Training Complete. Model saved to {final_path}")
        
        return model, test_df

    def evaluate(self, model_path=None, test_df=None, model_name="ppo_final_model"):
        print("--- Starting Evaluation ---")
        if model_path:
            model = PPO.load(model_path)
        else:
            final_path = self.output_dir / model_name
            if final_path.with_suffix(".zip").exists():
                 model = PPO.load(final_path)
            else:
                 raise FileNotFoundError(f"No model found to evaluate at {final_path}")

        if test_df is None:
             df = self.load_data()
             split_idx = int(len(df) * 0.8)
             test_df = df.iloc[split_idx:]

        # Use new Risk Env for Eval
        # We should ideally use the same enriched params as training, but defaults are strict enough for baseline eval
        env = TradingEnv(test_df)
        obs, _ = env.reset()
        done = False
        
        # Tracking
        equity_curve = [env.equity] # Start with initial balance
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            equity_curve.append(info['equity'])
            
        print(f"Final Equity: {equity_curve[-1]:.2f}")
        
        # Risk Evaluation
        from sampo2.agents.risk_evaluator import RiskEvaluator
        
        trade_log = env.trade_log if hasattr(env, 'trade_log') else []
        evaluator = RiskEvaluator(equity_curve, trade_log=trade_log)
        print(evaluator.get_report())
        
        # Save Trade Log
        if hasattr(env, 'trade_log') and env.trade_log:
            trades_df = pd.DataFrame(env.trade_log)
            if not trades_df.empty:
                trades_path = self.output_dir / "evaluation_trades.csv"
                trades_df.to_csv(trades_path, index=False)
                print(f"Trade log saved to {trades_path}")
            else:
                print("No trades occurred during evaluation.")
        
        return equity_curve

if __name__ == "__main__":
    trainer = PPOTrainer()
    model_name = "ppo_fresh_1m"
    trainer.train(total_timesteps=1000000, model_name=model_name)
    trainer.evaluate(model_name=model_name)
