
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback

class SB3ProgressCallback(BaseCallback):
    """
    A custom callback to display a TQDM progress bar during SB3 training.
    """
    def __init__(self, total_timesteps, description="Training"):
        super().__init__(verbose=0)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.description = description
        
    def _on_training_start(self) -> None:
        # Initialize the progress bar
        self.pbar = tqdm(total=self.total_timesteps, desc=self.description, unit="step")
        
    def _on_step(self) -> bool:
        # Update progress bar
        if self.pbar:
            self.pbar.update(self.locals['env'].num_envs) # usually 1, but safe
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
