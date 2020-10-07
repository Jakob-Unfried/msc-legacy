import jax.numpy as np

from typing import List


def make_schedule(step_size):
    """
    transforms various input types to x_1 step-size schedule
    """
    if type(step_size) is callable:
        return step_size
    if np.isscalar(step_size):
        return make_constant_schedule(step_size)
    raise ValueError(f'Unsupported type for `step_size`: {type(step_size)}')


def compose_schedules(schedules_list: List[callable], lengths_list: List[int], final_value=0.):
    """
    Use different schedules successively

    Parameters
    ----------
    schedules_list : List of callable
        A list of step size schedules
    lengths_list : List of int
        A list of lengths.
        `schedules_list[i]` is used for `lengths_list[i]` many indices.
        Note that the last of these can be `np.inf`
    final_value: float or None
        The value to use after all the specified schedules
        or None, then the last value of the last schedule is repeated infinitely

    Returns
    -------
    schedule : callable
        A step size schedule

    """
    assert len(schedules_list) == len(lengths_list)

    starting_points = [sum(lengths_list[:n]) for n in range(len(schedules_list))]

    def schedule(n):
        if n >= sum(lengths_list):
            if final_value is None:
                return schedules_list[-1](lengths_list[-1])
            else:
                return final_value

        for m in range(len(schedules_list)):
            if n < starting_points[m]:
                n_schedule = m - 1
                break
        else:
            n_schedule = -1

        return schedules_list[n_schedule](n - starting_points[n_schedule])

    return schedule


def make_constant_schedule(step, cutoff_idx=np.inf):
    """

    step - - - ▖▖▖▖▖▖▖▖▖▖▖▖

    0.0 - - - - - - - - - -▖▖▖▖▖▖
               |           |
               0           cutoff_idx
    """

    def schedule(n):
        if n >= cutoff_idx:
            return 0.
        return step

    return schedule


def make_linear_schedule(start_val, stop_val, stop_idx, cutoff_idx=np.inf):
    """
    start_val - - -▘▘▖▖
                       ▘▘▖▖
    stop_val - - - - - - - ▘▘▖▖▖▖▖▖▖▖▖▖▖

    0.0 - - - - - - - - - - - - - - - - ▖▖▖▖▖▖
                   |         |          |
                   0         stop_idx   cutoff_idx
    """

    def schedule(n):
        if n >= cutoff_idx:
            return 0.
        if n >= stop_idx:
            return stop_val
        t = n / stop_idx
        return start_val * (1 - t) + stop_val * t

    return schedule


def make_exponential_schedule(start_val, life_time_idx, cutoff_idx=np.inf, constant_offset=0.):
    """
    start_val          ▖
                       ▘▖
                        ▘▖
                          ▘▖▖
    constant_offset          ▘▘▘▖▖▖▖▖▖▖▖▖▖▖▖▖

    0.0                                      ▖▖▖▖▖▖▖▖▖▖▖▖▖▖
                       |                     |
                       0                     cutoff_idx
    """

    def schedule(n):
        if n >= cutoff_idx:
            return 0.
        return (start_val - constant_offset) * np.exp(- n / life_time_idx) + constant_offset

    return schedule
