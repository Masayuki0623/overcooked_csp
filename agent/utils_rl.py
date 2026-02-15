
from typing import Callable

def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return func
