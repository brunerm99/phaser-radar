# plotting.py

import numpy as np
from numpy.fft import fftshift
import plotly.graph_objects as go
import plotly.express as px

from utils import wrap


# def cfar_param_plot(
#     fig: go.Figure, x: np.ndarray, guard_cells: int, compute_cells: int, ref_index: int
# ):
#     """Plot CFAR parameters over existing plot.

#     Args:
#         fig (go.Figure): Figure to add regions to.
#         x (np.ndarray): Values of x-axis.
#         guard_cells (int): Number of guard cells.
#         compute_cells (int): Number of compute cells.
#         ref_index (int): Where to show cells surrounding.

#     Outputs:
#         fig (go.Figure): Updated figure.
#     """
#     guard_region = np.zeros(x.shape)
#     guard_region[np.arange(ref_index - guard_cells, ref_index + guard_cells)] = 1

#     compute_region = np.zeros(x.shape)
#     compute_region[
#         np.arange(ref_index - guard_cells - compute_cells, ref_index - guard_cells)
#     ] = 1
#     compute_region[
#         np.arange(ref_index + guard_cells, ref_index + guard_cells + compute_cells)
#     ] = 1

#     fig.add_trace(
#         go.Scatter(
#             x=wrap(x),
#             y=wrap(guard_region, fillval=0),
#             name="Guard Region",
#             fill="toself",
#             fillcolor="rgba(0,255,0,0.15)",
#             line_color="rgba(255,0,0,0)",
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=wrap(x),
#             y=wrap(compute_region, fillval=0),
#             name="Compute Region",
#             fill="toself",
#             fillcolor="rgba(0,0,255,0.15)",
#             line_color="rgba(0,0,0,0)",
#         )
#     )
#     return fig


def cfar_param_plot(
    fig, x, guard_cells, compute_cells, ref_index, min_val=0, max_val=1, shift=False
) -> go.Figure:
    """Plot CFAR parameters over existing plot.

    Args:
        fig (go.Figure): Figure to add regions to.
        x (np.ndarray): Values of x-axis.
        guard_cells (int): Number of guard cells.
        compute_cells (int): Number of compute cells.
        ref_index (int): Where to show cells surrounding.
        min_val (int, optional): Minimum value of signal / spectrum. Defaults to 0.
        max_val (int, optional): Maximum value of signal / spectrum. Defaults to 1.
        shift (bool, optional): Shift the FFT? Defaults to False.

    Returns:
        go.Figure: Updated figure.
    """
    guard_region = np.ones(x.shape) * min_val
    guard_region[np.arange(ref_index - guard_cells, ref_index + guard_cells)] = max_val

    compute_region = np.ones(x.shape) * min_val
    compute_region[
        np.arange(ref_index - guard_cells - compute_cells, ref_index - guard_cells)
    ] = max_val
    compute_region[
        np.arange(ref_index + guard_cells, ref_index + guard_cells + compute_cells)
    ] = max_val

    if shift:
        x_plot = fftshift(x)
    else:
        x_plot = x.copy()

    fig.add_trace(
        go.Scatter(
            x=wrap(x_plot),
            y=wrap(guard_region, fillval=min_val),
            name="Guard Region @ 100kHz",
            fill="toself",
            fillcolor="rgba(0,255,0,0.15)",
            line_color="rgba(255,0,0,0)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wrap(x_plot),
            y=wrap(compute_region, fillval=min_val),
            name="Compute Region @ 100kHz",
            fill="toself",
            fillcolor="rgba(0,0,255,0.15)",
            line_color="rgba(0,0,0,0)",
        )
    )
    return fig
