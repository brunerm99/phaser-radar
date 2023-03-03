# app.py

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from configparser import ConfigParser

from layouts import PLOTLY_DARK
from processing import cfar
from plotting import cfar_param_plot

# Pull in config
config = ConfigParser()
config.read("config.ini")

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

# Data
t = np.linspace(-8, 8, 1000)
sinc = np.sinc(t)

# Create app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dmc.NotificationsProvider(
    [
        dbc.Container(
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
                                        dmc.Text(id="guard-cells-text"),
                                        dmc.Slider(
                                            id="guard-cells-slider",
                                            min=guard_min,
                                            max=guard_max,
                                            step=guard_step,
                                            marks=[
                                                {"value": i, "label": str(i)}
                                                for i in range(
                                                    guard_min, guard_max + guard_step
                                                )[::10]
                                            ],
                                            value=guard_def,
                                        ),
                                        html.Br(),
                                        dmc.Text(id="compute-cells-text"),
                                        dmc.Slider(
                                            id="compute-cells-slider",
                                            min=compute_min,
                                            max=compute_max,
                                            step=compute_step,
                                            marks=[
                                                {"value": i, "label": str(i)}
                                                for i in range(
                                                    compute_min,
                                                    compute_max + compute_step,
                                                )[::10]
                                            ],
                                            value=compute_def,
                                        ),
                                        html.Br(),
                                        dmc.Text(id="bias-text"),
                                        dmc.Slider(
                                            id="bias-slider",
                                            min=bias_min,
                                            max=bias_max,
                                            step=bias_step,
                                            marks=[
                                                {"value": i, "label": str(i)}
                                                for i in range(
                                                    bias_min, bias_max + bias_step
                                                )
                                            ],
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
                                                        dmc.Button(
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
                                                        dmc.Button("Fetch new buffer"),
                                                    ],
                                                    style=dict(textAlign="center"),
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                # dbc.Row([html.Div([dmc.Button("What is CFAR?", id="open-modal"), style=dict(textAlign="center")])]),
                                html.Br(),
                                dbc.Row(
                                    [
                                        html.Div(
                                            [
                                                dmc.Button(
                                                    "What is CFAR?",
                                                    id="cfar-explanation-button",
                                                )
                                            ],
                                            style=dict(textAlign="center"),
                                        )
                                    ]
                                ),
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
                html.Div(id="notifications-container"),
                dmc.Modal(
                    title="What is CFAR?",
                    id="modal-simple",
                    children=[
                        dcc.Markdown(id="cfar-markdown", mathjax=True),
                        dmc.Space(h=20),
                        dmc.Group(
                            [
                                dmc.Button(
                                    "Close",
                                    color="red",
                                    variant="outline",
                                    id="modal-close-button",
                                ),
                            ],
                            position="right",
                        ),
                    ],
                    size="60%",
                ),
            ],
            className="pad-row",
            fluid=True,
        )
    ]
)


@app.callback(
    Output("modal-simple", "opened"),
    Output("cfar-markdown", "children"),
    Input("cfar-explanation-button", "n_clicks"),
    Input("modal-close-button", "n_clicks"),
    State("modal-simple", "opened"),
    prevent_initial_call=True,
)
def modal_demo(nc1, nc2, opened):
    with open("./static/markdown/cfar.md") as file:
        return not opened, file.read()
    return not opened


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
    ],
)
def update_cfar_plot(guard_cells, compute_cells, bias):
    window_mean = cfar(
        sinc,
        compute_cells=compute_cells,
        guard_cells=guard_cells,
        method=np.mean,
        bias=bias,
    )

    targets = sinc.copy()
    targets[np.where((targets < window_mean) | (np.isnan(window_mean)))] = np.nan

    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=t, y=sinc, name="Sinc", line=dict(color="rgba(0,127,255,0.5)"))
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=targets,
            name="Targets",
            line=dict(color="rgba(255,100,50,1)", width=5),
        )
    )
    fig.add_trace(go.Scatter(x=t, y=window_mean, name="CFAR"))
    fig.update_layout(PLOTLY_DARK)
    fig.update_layout(
        title="CFAR",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Amplitude"),
    )
    fig = cfar_param_plot(fig, t, guard_cells, compute_cells, ref_index=int(t.size / 2))
    return (
        fig,
        "Guard Cells: %i" % guard_cells,
        "Compute Cells: %i" % compute_cells,
        "Bias: %i" % bias,
    )


# TODO: Complete resetting to defaults
@app.callback(
    Output("notifications-container", "children"),
    Input("reset-default", "n_clicks"),
    prevent_initial_state=True,
)
def reset_default(n_clicks):
    print(n_clicks)
    if n_clicks is not None:
        return dmc.Notification(
            title="Hey there!",
            id="simple-notify",
            action="show",
            message="Notifications in Dash, Awesome!",
        )


if __name__ == "__main__":
    app.run_server(debug=True)
