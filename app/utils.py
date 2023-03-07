# utils.py

import numpy as np


def wrap(x: np.ndarray, fillval=None):
    """Wrap numpy array around itself for plotting of filled regions.

    Args:
        x (np.ndarray): Array to wrap.
        fillval (int|float): Value to wrap with. If None, use self backwards.

    Returns:
        wrapped_self (np.ndarray): Array wrapped.
    """
    if fillval is not None:
        return np.concatenate([x, np.ones(x.size) * fillval])
    return np.concatenate([x, x[::-1]])
