"""The monitoring dashboard.

Five tabs over the data the API logs. Reads the store directly rather than
going through HTTP, because the dashboard runs alongside the API and the extra
hop would buy nothing.

    python dashboard/app.py
    python dashboard/app.py --port 8050 --refresh 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.tabs import drift as drift_tab
from dashboard.tabs import overview as overview_tab
from dashboard.tabs import performance as performance_tab
from dashboard.tabs import registry as registry_tab
from dashboard.theme import (
    BACKGROUND,
    FONT,
    INK,
    LINE,
    MONO,
    MUTED,
    PANEL,
    TEAL,
    card,
    empty_state,
)
from monitoring.drift import build_evidently_report, compute_drift
from monitoring.performance import (
    daily_metrics,
    load_predictions,
    segment_metrics,
    snapshot,
    traffic_summary,
)
from serving.store import PredictionStore, StoreUnavailableError
from src.config import load_config
from src.logger import get_logger, setup_logging
from tracking.mlflow_tracking import get_tracker

logger = get_logger(__name__)

DEFAULT_PORT = 8050
DEFAULT_REFRESH = 30
DEFAULT_WINDOW = 90

TABS = [
    ("overview", "Overview"),
    ("performance", "Performance"),
    ("drift", "Drift"),
    ("registry", "Models"),
    ("about", "About"),
]

config = load_config(create_dirs=False)
store = PredictionStore()
tracker = get_tracker()

app = Dash(__name__, title="Freight Rate Monitoring", suppress_callback_exceptions=True)
server = app.server


def header() -> html.Div:
    """The bar across the top.

    Returns:
        The header.
    """
    return html.Div([
        html.Div([
            html.Div("Freight Rate Monitoring", style={
                "fontSize": "19px", "fontWeight": 700, "color": INK,
            }),
            html.Div(
                "Live model performance, drift and training history",
                style={"fontSize": "12.5px", "color": MUTED, "marginTop": "2px"},
            ),
        ]),
        html.Div(id="header-status", style={"textAlign": "right"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "18px 26px", "background": PANEL,
        "borderBottom": f"1px solid {LINE}",
    })


app.layout = html.Div([
    dcc.Interval(id="refresh", interval=DEFAULT_REFRESH * 1000, n_intervals=0),
    dcc.Store(id="window-days", data=DEFAULT_WINDOW),
    header(),
    html.Div(
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[dcc.Tab(label=label, value=value) for value, label in TABS],
            style={"borderBottom": f"1px solid {LINE}"},
        ),
        style={"background": PANEL, "padding": "0 14px"},
    ),
    dcc.Loading(
        html.Div(id="tab-content", style={"padding": "22px 26px", "maxWidth": "1400px"}),
        type="dot",
        color=TEAL,
    ),
], style={
    "fontFamily": FONT, "background": BACKGROUND, "minHeight": "100vh", "color": INK,
})


@app.callback(
    Output("header-status", "children"),
    Input("refresh", "n_intervals"),
)
def update_header(_) -> html.Div:
    """Show whether the store and MLflow are reachable.

    Args:
        _: The refresh tick, unused.

    Returns:
        The status block.
    """
    parts = []

    try:
        count = store.count_predictions()
        parts.append(f"{count:,} predictions logged")
    except StoreUnavailableError:
        parts.append("store unavailable")

    parts.append("MLflow connected" if tracker.is_available else "MLflow not connected")

    return html.Div(
        " · ".join(parts),
        style={"fontSize": "12px", "color": MUTED, "fontFamily": MONO},
    )


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("refresh", "n_intervals"),
    State("window-days", "data"),
)
def render_tab(tab: str, _, days: int) -> html.Div:
    """Build whichever tab is selected.

    Args:
        tab: The selected tab.
        _: The refresh tick, unused.
        days: Size of the window to report on.

    Returns:
        The tab content.
    """
    if not store.is_available:
        store.connect()

    try:
        if tab == "overview":
            return overview_tab.render(
                snapshot(store, days).to_dict(),
                traffic_summary(store, days),
                compute_drift(store, config, days).to_dict(),
                daily_metrics(store, days),
            )

        if tab == "performance":
            return performance_tab.render(
                snapshot(store, days).to_dict(),
                segment_metrics(store, "equipment", days),
                segment_metrics(store, "distance_band", days),
                daily_metrics(store, days),
            )

        if tab == "drift":
            return drift_tab.render(
                compute_drift(store, config, days).to_dict(),
                _daily_unknown_city(days),
            )

        if tab == "registry":
            return registry_tab.render(
                _serving_model_info(),
                tracker.recent_runs(),
                tracker.model_versions(),
                tracker,
            )

        return _about()

    except StoreUnavailableError as exc:
        return card(empty_state(
            "Could not reach the prediction store.",
            f"{exc}. Check DATABASE_URL, or start the database with docker compose up.",
        ))
    except Exception as exc:
        logger.exception("Failed to render the %s tab", tab)
        return card(empty_state(f"Could not build this tab: {exc}"))


def _daily_unknown_city(days: int):
    """Work out the unknown city rate per day.

    Args:
        days: How far back to look.

    Returns:
        One row per day, or None when there is no traffic.
    """
    frame = load_predictions(store, days)

    if frame.empty:
        return None

    frame["day"] = frame["predicted_at"].dt.date
    return (
        frame.groupby("day")["unknown_city"]
        .mean()
        .reset_index(name="unknown_city_rate")
        .sort_values("day")
    )


def _serving_model_info() -> dict:
    """Read the metadata of the model the API is serving.

    Returns:
        The metadata, flattened, or an empty dict when it cannot be read.
    """
    import json

    path = config.paths.metadata_file

    if not path.is_file():
        return {}

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return {
        "trained_at": meta.get("trained_at", ""),
        "estimator": meta.get("model", {}).get("estimator", ""),
        "n_features": meta.get("features", {}).get("n_features", 0),
        "known_cities": meta.get("features", {}).get("n_known_cities", 0),
        "training_rows": meta.get("training", {}).get("n_rows", 0),
        "training_from": meta.get("training", {}).get("date_from", ""),
        "training_to": meta.get("training", {}).get("date_to", ""),
        "validation": meta.get("validation", {}),
    }


@app.callback(
    Output("evidently-output", "children"),
    Input("build-evidently", "n_clicks"),
    State("window-days", "data"),
    prevent_initial_call=True,
)
def build_evidently(clicks: int, days: int) -> html.Div:
    """Build the full Evidently report on demand.

    Args:
        clicks: How many times the button has been pressed.
        days: Size of the window to report on.

    Returns:
        The report in an iframe, or an explanation of why it could not be built.
    """
    if not clicks:
        return html.Div()

    output = config.paths.figures_dir.parent / "evidently" / "data_drift.html"
    html_report = build_evidently_report(store, config, days, output)

    if not html_report:
        return html.Div(
            "Could not build the report. There may be too little traffic, or "
            "Evidently may not be installed.",
            style={"fontSize": "13px", "color": MUTED},
        )

    return html.Div([
        html.Div(
            f"Saved to {output}",
            style={"fontSize": "12px", "color": MUTED, "marginBottom": "10px",
                   "fontFamily": MONO},
        ),
        html.Iframe(
            srcDoc=html_report,
            style={"width": "100%", "height": "900px", "border": f"1px solid {LINE}",
                   "borderRadius": "8px", "background": PANEL},
        ),
    ])


def _about() -> html.Div:
    """Explain what the dashboard is showing and why.

    Returns:
        The about tab.
    """
    from dashboard.theme import section

    return html.Div([
        card([
            section("What this is"),
            html.Div([
                html.P(
                    "A freight rate model in production, with the monitoring that "
                    "goes around it. The model prices a load from its lane, "
                    "distance, trailer type, weight and date.",
                    style={"lineHeight": "1.65", "marginTop": 0},
                ),
                html.P([
                    ("Two things about this problem make the monitoring worth "
                    "looking at. "),
                    html.B("Outcomes arrive late"),
                    (": a rate is quoted now and confirmed days or weeks later, so "
                    "accuracy can only be measured on the traffic that has come "
                    "back. Coverage is reported next to every metric for that "
                    "reason. "),
                    html.B("The model cannot see the period it predicts"),
                    (": training stops on 31 October and every load priced "
                    "afterwards is a forecast, so drift is not hypothetical."),
                ], style={"lineHeight": "1.65"}),
            ], style={"fontSize": "13.5px", "color": INK}),
        ]),
        card([
            section("The tabs"),
            html.Ul([
                html.Li([html.B("Overview"), " — volume, accuracy and drift status."]),
                html.Li([html.B("Performance"), (" — live error against the validation "
                         "holdout, split by trailer type and distance.")]),
                html.Li([html.B("Drift"), (" — how far traffic has moved from the "
                         "training data, with the full Evidently report on demand.")]),
                html.Li([html.B("Models"), (" — the serving model, the registry and "
                         "every training run, linking out to MLflow.")]),
            ], style={"fontSize": "13.5px", "lineHeight": "1.9", "paddingLeft": "20px"}),
        ]),
        card([
            section("Running it"),
            html.Div([
                "docker compose up --build", html.Br(),
                "python simulator/replay.py --speed 200",
            ], style={
                "fontFamily": MONO, "fontSize": "12.5px", "background": BACKGROUND,
                "padding": "14px 16px", "borderRadius": "7px", "lineHeight": "1.9",
            }),
            html.Div(
                "The replay streams the validation set at the API day by day. "
                "Outcomes in that replay are synthetic and labelled as such.",
                style={"fontSize": "12.5px", "color": MUTED, "marginTop": "12px",
                       "lineHeight": "1.6"},
            ),
        ]),
    ])


def main() -> int:
    """Start the dashboard.

    Returns:
        0 on a clean exit.
    """
    parser = argparse.ArgumentParser(description="Freight rate monitoring dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--refresh", type=int, default=DEFAULT_REFRESH,
                        help="seconds between refreshes")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")
    store.connect()

    logger.info("Dashboard on http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())