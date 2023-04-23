# app.py

from dash import Dash, html, dcc, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numpy.lib.stride_tricks import sliding_window_view
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from configparser import ConfigParser

from layouts import PLOTLY_DARK
from processing import cfar
from plotting import cfar_param_plot
from phaser import setup, tx, rx

# Pull in config
config = ConfigParser()
config.read("config.ini")
phaser_config = dict(config["phaser"])

bias_min = int(config["user_inputs"]["bias_min"])
bias_max = int(config["user_inputs"]["bias_max"])
bias_step = int(config["user_inputs"]["bias_step"])
bias_def = int(config["user_inputs"]["bias_def"])
compute_min = int(config["user_inputs"]["compute_min"])
compute_max = int(config["user_inputs"]["compute_max"])
compute_step = int(config["user_inputs"]["compute_step"])
compute_def = int(config["user_inputs"]["compute_def"])
guard_min = int(config["user_inputs"]["guard_min"])
guard_max = int(config["user_inputs"]["guard_max"])
guard_step = int(config["user_inputs"]["guard_step"])
guard_def = int(config["user_inputs"]["guard_def"])

# Collect data
my_sdr, my_phaser, N, ts = setup(phaser_config)
try:
    tx(my_sdr, phaser_config)
except:
    pass
signal, signal_fft, freq = rx(my_sdr)

# Dummy data
# t = np.linspace(-8, 8, 1000)
# signal_fft = np.sinc(t)
# freq = np.arange(signal_fft.size)

# Create app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(children="CFAR Example"),
                        html.Div(
                            children="This is showing how to perform and modify the CFAR algorithm."
                        ),
                    ]
                )
            ],
            align="end",
        ),
        html.Hr(),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Col([html.H5("Input parameters")]),
                                html.P(id="guard-cells-text"),
                                dcc.Slider(
                                    guard_min,
                                    guard_max,
                                    id="guard-cells-slider",
                                    step=guard_step,
                                    marks={
                                        f"{i}": f"{i}"
                                        for i in range(
                                            guard_min, guard_max + guard_step
                                        )[:: guard_step * 2]
                                    },
                                    value=guard_def,
                                ),
                                html.Br(),
                                html.P(id="compute-cells-text"),
                                dcc.Slider(
                                    compute_min,
                                    compute_max,
                                    id="compute-cells-slider",
                                    step=compute_step,
                                    marks={
                                        f"{i}": f"{i}"
                                        for i in range(
                                            compute_min, compute_max + compute_step
                                        )[:: compute_step * 2]
                                    },
                                    value=compute_def,
                                ),
                                html.Br(),
                                html.P(id="bias-text"),
                                dcc.Slider(
                                    bias_min,
                                    bias_max,
                                    id="bias-slider",
                                    step=bias_step,
                                    marks={
                                        f"{i}": f"{i}"
                                        for i in range(bias_min, bias_max + bias_step)
                                    },
                                    value=bias_def,
                                ),
                                html.Br(),
                            ]
                        ),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Reset to defaults",
                                                    id="reset-default",
                                                ),
                                            ],
                                            style=dict(textAlign="center"),
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Fetch new buffer",
                                                    id="fetch-new-buffer",
                                                ),
                                            ],
                                            style=dict(textAlign="center"),
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        html.Br(),
                    ],
                    width=3,
                ),
                # Plot
                dbc.Col(
                    [
                        dcc.Graph(id="cfar-plot", style={"height": "60vh"}),
                    ],
                    width=True,
                    style=dict(height="100%"),
                ),
            ],
        ),
    ],
    className="pad-row",
    fluid=True,
)


@app.callback(
    [
        Output("cfar-plot", "figure"),
        Output("guard-cells-text", "children"),
        Output("compute-cells-text", "children"),
        Output("bias-text", "children"),
    ],
    [
        Input("guard-cells-slider", "value"),
        Input("compute-cells-slider", "value"),
        Input("bias-slider", "value"),
        Input("fetch-new-buffer", "n_clicks"),
    ],
)
def update_cfar_plot(guard_cells, compute_cells, bias, fetch_new_buffer):
    if "fetch-new-buffer" == ctx.triggered_id:
        print("Fetching new buffer")
        try:
            tx(my_sdr, phaser_config)
        except:
            pass
        _, signal_fft2, freq2 = rx(my_sdr)
        window_mean = cfar(
            signal_fft2,
            compute_cells=compute_cells,
            guard_cells=guard_cells,
            method=np.mean,
            bias=bias,
        )

        targets = signal_fft2.copy()
        targets[np.where((targets < window_mean) | (np.isnan(window_mean)))] = np.nan

        fig = make_subplots()
        fig.add_trace(
            go.Scatter(
                x=freq2,
                y=fftshift(np.log10(signal_fft2)),
                name="Signal FFT",
                line=dict(color="rgba(0,127,255,0.5)"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=freq2,
                y=fftshift(np.log10(targets)),
                name="Targets",
                line=dict(color="rgba(255,100,50,1)", width=5),
            )
        )
        fig.add_trace(
            go.Scatter(x=freq2, y=fftshift(np.log10(window_mean)), name="CFAR")
        )
        fig.update_layout(PLOTLY_DARK)
        fig.update_layout(
            title="CFAR",
            xaxis=dict(title="Frequency (Hz)", range=[100e3, 140e3]),
            yaxis=dict(title="Amplitude"),
        )
        ref_index = np.argmin(
            np.abs(freq2 - float(phaser_config["signal_freq_mhz"]) * 1e6)
        )
        fig = cfar_param_plot(
            fig,
            freq2,
            guard_cells,
            compute_cells,
            ref_index=ref_index,
            min_val=np.log10(signal_fft2).min(),
            max_val=np.log10(signal_fft2).max(),
            shift=False,
        )
        return (
            fig,
            "Guard Cells: %i" % guard_cells,
            "Compute Cells: %i" % compute_cells,
            "Bias: %i" % bias,
        )

    window_mean = cfar(
        signal_fft,
        compute_cells=compute_cells,
        guard_cells=guard_cells,
        method=np.mean,
        bias=bias,
    )

    targets = signal_fft.copy()
    targets[np.where((targets < window_mean) | (np.isnan(window_mean)))] = np.nan

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=fftshift(np.log10(signal_fft)),
            name="Signal FFT",
            line=dict(color="rgba(0,127,255,0.5)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=fftshift(np.log10(targets)),
            name="Targets",
            line=dict(color="rgba(255,100,50,1)", width=5),
        )
    )
    fig.add_trace(go.Scatter(x=freq, y=fftshift(np.log10(window_mean)), name="CFAR"))
    fig.update_layout(PLOTLY_DARK)
    fig.update_layout(
        title="CFAR",
        xaxis=dict(title="Frequency (Hz)", range=[100e3, 140e3]),
        yaxis=dict(title="Amplitude"),
    )
    ref_index = np.argmin(np.abs(freq - float(phaser_config["signal_freq_mhz"]) * 1e6))
    fig = cfar_param_plot(
        fig,
        freq,
        guard_cells,
        compute_cells,
        ref_index=ref_index,
        min_val=np.log10(signal_fft).min(),
        max_val=np.log10(signal_fft).max(),
        shift=False,
    )
    return (
        fig,
        "Guard Cells: %i" % guard_cells,
        "Compute Cells: %i" % compute_cells,
        "Bias: %i" % bias,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
