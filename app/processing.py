# processing.py

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def cfar(
    x: np.ndarray, compute_cells: int, guard_cells: int, bias: float = 1, method=np.mean
):
    """Computes CFAR threshold.

    Args:
        x (np.ndarray): Input array to compute threshold for.
        compute_cells (int): Cells surrounding reference cell to compute with.
        guard_cells (int): Cells surrounding reference cell to discard.
        bias (float, optional): Multiply threshold by this amount. Defaults to 1.
        method (function, optional): Method to compute over compute_cells. Defaults to np.mean.

    Returns:
        np.ndarray: CFAR threshold.
    """
    pad = int((compute_cells + guard_cells))
    # fmt: off
    window_mean = np.pad(                                                               # Pad front/back since n_windows < n_points
        method(                                                                         # Apply input method to remaining compute cells
            np.delete(                                                                  # Remove guard cells, CUT from computation
                sliding_window_view(x, (compute_cells * 2) + (guard_cells * 2)),        # Windows of x including CUT, guard cells, and compute cells
                np.arange(int(compute_cells), compute_cells + (guard_cells * 2) + 1),   # Get indices of guard cells, CUT
                axis=1), 
            axis=1
        ), (pad - 1, pad),                                                               
        constant_values=(np.nan, np.nan)                                                # Fill with NaNs
    ) * bias                                                                            # Multiply output by bias over which cell is not noise
    # fmt: on
    return window_mean
