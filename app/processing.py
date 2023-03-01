# processing.py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def cfar(
    x: np.ndarray, compute_cells: int, guard_cells: int, bias: float = 1, method=np.mean
):
    pad = int((compute_cells + guard_cells))
    window_mean = (
        np.pad(  # Pad front/back since n_windows < n_points
            method(  # Take mean of remaining compute cells
                np.delete(  # Remove guard cells, CUT from computation
                    sliding_window_view(
                        x, (compute_cells * 2) + (guard_cells * 2)
                    ),  # Windows of x including CUT, guard cells, and compute cells
                    np.arange(
                        int(compute_cells), compute_cells + (guard_cells * 2) + 1
                    ),  # Get indices of guard cells, CUT
                    axis=1,
                ),
                axis=1,
            ),
            (pad - 1, pad),
        )
        * bias  # Multiply output by bias over which cell is not noise
    )
    return window_mean


def wrap(x, fillval=None):
    if fillval is not None:
        return np.concatenate([x, np.ones(x.size) * fillval])
    return np.concatenate([x, x[::-1]])
