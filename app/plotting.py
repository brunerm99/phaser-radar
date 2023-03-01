# plotting.py

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from helpers import wrap


def cfar_param_plot(
    fig: go.Figure, x: np.ndarray, guard_cells: int, compute_cells: int, ref_index: int
):
    """Plot CFAR parameters over existing plot.

    Args:
        fig (go.Figure): Figure to add regions to.
        x (np.ndarray): Values of x-axis.
        guard_cells (int): Number of guard cells.
        compute_cells (int): Number of compute cells.
        ref_index (int): Where to show cells surrounding.

    Outputs:
        fig (go.Figure): Updated figure.
    """
    guard_region = np.zeros(x.shape)
    guard_region[np.arange(ref_index - guard_cells, ref_index + guard_cells)] = 1

    compute_region = np.zeros(x.shape)
    compute_region[
        np.arange(ref_index - guard_cells - compute_cells, ref_index - guard_cells)
    ] = 1
    compute_region[
        np.arange(ref_index + guard_cells, ref_index + guard_cells + compute_cells)
    ] = 1

    fig.add_trace(
        go.Scatter(
            x=wrap(x),
            y=wrap(guard_region, fillval=0),
            name="Guard Region",
            fill="toself",
            fillcolor="rgba(0,255,0,0.15)",
            line_color="rgba(255,0,0,0)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wrap(x),
            y=wrap(compute_region, fillval=0),
            name="Compute Region",
            fill="toself",
            fillcolor="rgba(0,0,255,0.15)",
            line_color="rgba(0,0,0,0)",
        )
    )
    return fig
