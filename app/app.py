# app.py

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from layouts import PLOTLY_DARK
from processing import cfar
from plotting import cfar_param_plot

# Data
t = np.linspace(-8, 8, 1000)
sinc = np.sinc(t)

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
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Col([html.H5("Input parameters")]),
                                html.P("Guard Cells"),
                                dcc.Slider(
                                    id="guard-cells-slider",
                                    min=0,
                                    max=100,
                                    marks={i: str(i) for i in range(101)[::5]},
                                    value=60,
                                ),
                                html.P("Compute Cells"),
                                dcc.Slider(
                                    id="compute-cells-slider",
                                    min=0,
                                    max=100,
                                    marks={i: str(i) for i in range(101)[::5]},
                                    value=80,
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.Div([html.Button("Fetch buffer")]),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="cfar-plot", style={"height": "60vh"}),
                    ],
                    width=True,
                ),
            ]
        ),
        html.Hr(),
        html.P([html.A("Google", href="google.com")]),
    ],
    fluid=True,
)


@app.callback(
    Output("cfar-plot", "figure"),
    Input("guard-cells-slider", "value"),
    Input("compute-cells-slider", "value"),
)
def update_cfar_plot(guard_cells, compute_cells):  # guard_cells, compute_cells):
    window_mean = cfar(
        sinc,
        compute_cells=compute_cells,
        guard_cells=guard_cells,
        method=np.mean,
        bias=2,
    )
    fig = make_subplots()
    fig.add_trace(go.Scatter(x=t, y=sinc, name="Sinc"))
    fig.add_trace(go.Scatter(x=t, y=window_mean, name="CFAR"))
    fig.update_layout(PLOTLY_DARK)
    fig.update_layout(
        title="CFAR",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Amplitude"),
    )
    cfar_param_plot(fig, t, guard_cells, compute_cells, ref_index=int(t.size / 2))
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
