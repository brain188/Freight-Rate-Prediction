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
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.charts import error_over_time, predicted_vs_actual_series
from dashboard.tabs import drift as drift_tab
from dashboard.tabs import overview as overview_tab
from dashboard.tabs import performance as performance_tab
from dashboard.tabs import registry as registry_tab
from dashboard.theme import (
    ALERT,
    BACKGROUND,
    FONT,
    INK,
    LINE,
    MONO,
    MUTED,
    OK,
    PANEL,
    TEAL,
    UNKNOWN,
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
DEFAULT_REFRESH = 5
STRUCTURE_REFRESH = 300
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

# Loaded once and kept. It is 47,000 rows and never changes while the process
# runs, so reloading it on a five second refresh would be wasteful.
_reference: pd.DataFrame | None = None

# Last known drift status, so the fast refresh can show it without paying for
# a full comparison against the training data every few seconds.
_cached_drift_status = "unknown"


def reference_frame() -> pd.DataFrame | None:
    """Return the cleaned training data used as the drift reference.

    Returns:
        The training loads, or None when they cannot be read.
    """
    global _reference

    if _reference is None:
        try:
            from monitoring.drift import reference_data

            _reference = reference_data(config)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.warning(
                "Could not load the training data for drift comparison: %s",
                exc,
            )
            _reference = pd.DataFrame()

    return _reference if not _reference.empty else None

app = Dash(__name__, title="Freight Rate Monitoring", suppress_callback_exceptions=True)
server = app.server


def header() -> html.Div:
    """The bar across the top.

    Returns:
        The header.
    """
    return html.Div([
        html.Div([
            html.Div("F", style={
                "width": "34px", "height": "34px", "borderRadius": "9px",
                "background": TEAL, "color": "#FFFFFF", "display": "flex",
                "alignItems": "center", "justifyContent": "center",
                "fontSize": "15px", "fontWeight": 800, "fontFamily": MONO,
                "marginRight": "13px", "flexShrink": 0,
                "boxShadow": "0 2px 6px rgba(6, 74, 86, 0.32)",
            }),
            html.Div([
                html.Div("Freight Rate Monitoring", style={
                    "fontSize": "16.5px", "fontWeight": 700, "color": INK,
                    "letterSpacing": "-0.01em",
                }),
                html.Div(
                    "Live model performance, drift and training history",
                    style={"fontSize": "12px", "color": MUTED, "marginTop": "2px"},
                ),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div(id="header-status", style={"textAlign": "right"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "15px 26px", "background": PANEL,
        "borderBottom": f"1px solid {LINE}",
    })


def _header_divider() -> html.Div:
    """A short vertical rule between items in the status strip.

    Returns:
        The divider.
    """
    return html.Div(style={
        "width": "1px", "height": "14px", "background": LINE, "margin": "0 16px",
    })


def _header_stat(label: str, value: str) -> html.Div:
    """One "label value" pair in the status strip, value in monospace.

    Args:
        label: What the number measures.
        value: The number, already formatted.

    Returns:
        The stat block.
    """
    return html.Div([
        html.Span(label, style={"color": MUTED, "marginRight": "6px"}),
        html.Span(value, style={"color": INK, "fontFamily": MONO, "fontWeight": 600}),
    ], style={"fontSize": "12px", "display": "flex", "alignItems": "center"})


def _tab_style(selected: bool) -> dict:
    """Style for one entry in the tab strip.

    Dash's default tab CSS reads like a stack of grey boxes. An inline style
    always wins the cascade, so a plain underline is drawn here instead,
    closer to how a data tool like this should look.

    Args:
        selected: Whether this is the active tab.

    Returns:
        The style dict for dcc.Tab.
    """
    base = {
        "padding": "13px 2px", "marginRight": "30px", "border": "none",
        "borderBottom": "2px solid transparent", "background": "transparent",
        "fontSize": "13px", "fontWeight": 600, "color": MUTED,
        "letterSpacing": "0.1px",
    }
    if selected:
        base.update({"color": TEAL, "borderBottom": f"2px solid {TEAL}"})
    return base


app.layout = html.Div([
    dcc.Interval(id="refresh", interval=DEFAULT_REFRESH * 1000, n_intervals=0),
    dcc.Interval(id="slow-refresh", interval=STRUCTURE_REFRESH * 1000, n_intervals=0),
    dcc.Store(id="window-days", data=DEFAULT_WINDOW),
    header(),
    html.Div(
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(
                    label=label, value=value,
                    style=_tab_style(False), selected_style=_tab_style(True),
                )
                for value, label in TABS
            ],
            style={"height": "auto"},
        ),
        style={"background": PANEL, "padding": "0 26px", "borderBottom": f"1px solid {LINE}"},
    ),
    dcc.Loading(
        html.Div(id="tab-content", style={"padding": "22px 26px", "maxWidth": "1400px"}),
        type="dot",
        color=TEAL,
        delay_show=400,
        delay_hide=200,
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
    try:
        count_text = f"{store.count_predictions():,}"
        store_ok = True
    except StoreUnavailableError:
        count_text = "—"
        store_ok = False

    if store_ok:
        live_pill = html.Div([
            html.Span(className="live-dot"),
            html.Span("LIVE", style={
                "marginLeft": "8px", "fontSize": "10.5px", "fontWeight": 700,
                "letterSpacing": "0.6px", "color": OK,
            }),
        ], style={
            "display": "flex", "alignItems": "center",
            "background": "#E7F6EC", "padding": "4px 11px 4px 9px",
            "borderRadius": "20px",
        })
    else:
        live_pill = html.Div([
            html.Span(style={
                "width": "7px", "height": "7px", "borderRadius": "50%",
                "background": ALERT, "display": "inline-block",
            }),
            html.Span("OFFLINE", style={
                "marginLeft": "8px", "fontSize": "10.5px", "fontWeight": 700,
                "letterSpacing": "0.6px", "color": ALERT,
            }),
        ], style={
            "display": "flex", "alignItems": "center",
            "background": "#FBECEA", "padding": "4px 11px 4px 9px",
            "borderRadius": "20px",
        })

    mlflow_dot = html.Div([
        html.Span("MLflow", style={"color": MUTED, "marginRight": "7px"}),
        html.Span("●", style={
            "color": OK if tracker.is_available else UNKNOWN, "fontSize": "9px",
        }),
    ], style={"fontSize": "12px", "display": "flex", "alignItems": "center"})

    timestamp = datetime.now(UTC).strftime("%H:%M:%S")

    return html.Div([
        live_pill, _header_divider(),
        _header_stat("predictions", count_text), _header_divider(),
        mlflow_dot, _header_divider(),
        _header_stat("updated", f"{timestamp} UTC"),
    ], style={"display": "flex", "alignItems": "center"})


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("slow-refresh", "n_intervals"),
    State("window-days", "data"),
)
def render_tab(tab: str, _, days: int) -> html.Div:
    """Build whichever tab is selected.

    Fires on a tab change and on a slow timer, never on the fast one. The fast
    interval updates figures in place instead, so the page does not blank while
    new data is fetched.

    Args:
        tab: The selected tab.
        _: The slow refresh tick, unused.
        days: Size of the window to report on.

    Returns:
        The tab content.
    """
    if not store.is_available:
        store.connect()

    try:
        if tab == "overview":
            drift_now = compute_drift(store, config, days).to_dict()
            _remember_drift_status(drift_now["status"])
            return overview_tab.render(
                snapshot(store, days).to_dict(),
                traffic_summary(store, days),
                drift_now,
                daily_metrics(store, days),
                load_predictions(store, days),
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
                load_predictions(store, days),
                reference_frame(),
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


@app.callback(
    Output("live-predicted-actual", "figure"),
    Output("live-error", "figure"),
    Output("live-headline-wrap", "children"),
    Input("refresh", "n_intervals"),
    State("tabs", "value"),
    State("window-days", "data"),
    prevent_initial_call=True,
)
def update_live_charts(_, tab: str, days: int):
    """Refresh the overview figures without rebuilding the page.

    Only the figure data is sent, so Plotly patches the existing charts in
    place. Nothing blanks, scroll position holds, and a chart the user is
    hovering over keeps its tooltip.

    Args:
        _: The refresh tick, unused.
        tab: Which tab is showing.
        days: Size of the window to report on.

    Returns:
        The two figures and the headline block.

    Raises:
        PreventUpdate: When the overview is not the visible tab, or the store
            cannot be reached.
    """
    if tab != "overview":
        raise PreventUpdate

    try:
        # Read once and derive everything from it. The tab used to run four
        # separate queries per refresh, and on a five second timer that is the
        # difference between a page that keeps up and one that does not.
        frame = load_predictions(store, days)

        if frame.empty:
            raise PreventUpdate

        return (
            predicted_vs_actual_series(frame),
            error_over_time(frame),
            overview_tab.headline(*_live_headline_inputs(frame, days)),
        )
    except PreventUpdate:
        raise
    except Exception as exc:
        logger.debug("Live refresh skipped", exc_info=True)
        raise PreventUpdate from exc


def _live_headline_inputs(frame, days: int) -> tuple[dict, dict, dict]:
    """Derive the headline numbers from an already loaded frame.

    Drift status is deliberately not recomputed here. It compares against
    47,000 training rows, changes slowly, and is refreshed by the slow timer
    and by the drift tab itself.

    Args:
        frame: Logged predictions already read from the store.
        days: Size of the window being reported on.

    Returns:
        Performance, traffic and drift dictionaries for the headline.
    """
    import numpy as np

    scored = frame[frame["actual_rate"].notna()]

    performance = {
        "n_predictions": len(frame),
        "n_scored": len(scored),
        "coverage": len(scored) / len(frame) if len(frame) else 0.0,
        "metrics": {},
    }

    if not scored.empty:
        error = (scored["actual_rate"] - scored["predicted_rate"]).to_numpy()
        actual = scored["actual_rate"].to_numpy()
        performance["metrics"] = {
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "mape": float(np.mean(np.abs(error / actual)) * 100),
            "bias": float(-np.mean(error)),
        }

    traffic = {
        "window_days": days,
        "n_predictions": len(frame),
        "unknown_city_rate": float(frame["unknown_city"].mean()),
    }

    return performance, traffic, {"status": _cached_drift_status}


def _remember_drift_status(status: str) -> None:
    """Keep the latest drift status for the fast refresh to reuse.

    Args:
        status: The status just computed.
    """
    global _cached_drift_status
    _cached_drift_status = status


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
