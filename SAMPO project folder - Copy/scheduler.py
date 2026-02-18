
from typing import Callable

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear schedule for learning rate or entropy coef.
    
    :param initial_value: The initial value (at progress_remaining = 1.0)
    :param final_value: The final value (at progress_remaining = 0.0)
    :return: schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining starts at 1.0 and goes to 0.0.
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return func
