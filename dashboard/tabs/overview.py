"""The overview tab: is the model healthy right now.

Answers three questions in one screen. How much traffic is coming through,
whether accuracy has moved, and whether traffic still resembles what the model
was trained on. Everything else is a tab away.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from dashboard.charts import error_over_time, predicted_vs_actual_series
from dashboard.theme import (
    ALERT,
    CHART_LAYOUT,
    INK,
    MUTED,
    OK,
    TEAL,
    WARN,
    badge,
    card,
    empty_state,
    metric,
    row,
    section,
)


def _accuracy_colour(mape: float) -> str:
    """Colour a MAPE by how far it has drifted from the holdout figure.

    The model scored 1.88 percent on the holdout, so live traffic much above
    that is worth noticing.

    Args:
        mape: Mean absolute percentage error on live traffic.

    Returns:
        A colour.
    """
    if mape <= 3.0:
        return OK
    if mape <= 5.0:
        return WARN
    return ALERT


def render(performance: dict, traffic: dict, drift: dict, daily, frame=None) -> html.Div:
    """Build the overview.

    Args:
        performance: Snapshot from the performance endpoint.
        traffic: Summary of recent traffic.
        drift: The drift report.
        daily: Per day metrics as a dataframe.
        frame: Raw logged predictions, used for the live series.

    Returns:
        The tab content.
    """
    if not traffic.get("n_predictions"):
        return card(
            empty_state(
                "No predictions logged yet.",
                "python simulator/replay.py --speed 200",
            )
        )

    metrics = performance.get("metrics", {})
    scored = performance.get("n_scored", 0)

    return html.Div(
        [
            html.Div(
                _headline(performance, traffic, drift, metrics, scored), id="live-headline-wrap"
            ),
            _live_series(frame),
            _volume_chart(daily),
            _accuracy_chart(daily),
            _signals(traffic, drift),
        ]
    )


def headline(performance: dict, traffic: dict, drift: dict) -> html.Div:
    """Build just the headline numbers, for the interval callback to swap in.

    Args:
        performance: Performance snapshot.
        traffic: Traffic summary.
        drift: Drift report.

    Returns:
        The headline row.
    """
    metrics = performance.get("metrics", {})
    return _headline(performance, traffic, drift, metrics, performance.get("n_scored", 0))


def _headline(performance, traffic, drift, metrics, scored) -> html.Div:
    """The four numbers at the top.

    Args:
        performance: Performance snapshot.
        traffic: Traffic summary.
        drift: Drift report.
        metrics: Metric values from the snapshot.
        scored: How many predictions could be scored.

    Returns:
        The headline block.
    """
    if scored:
        mape = metrics.get("mape", 0)
        accuracy_value = f"{mape:.2f}%"
        accuracy_note = f"vs 1.88% on the holdout · {scored:,} scored"
        accuracy_colour = _accuracy_colour(mape)
    else:
        accuracy_value = "waiting"
        accuracy_note = "no outcomes reported yet"
        accuracy_colour = MUTED

    return card(
        [
            row(
                [
                    metric(
                        "Predictions",
                        f"{traffic['n_predictions']:,}",
                        f"last {traffic['window_days']} days",
                    ),
                    metric("Live accuracy (MAPE)", accuracy_value, accuracy_note, accuracy_colour),
                    metric(
                        "Coverage",
                        f"{performance.get('coverage', 0):.0%}",
                        "share with a confirmed rate",
                    ),
                    metric(
                        "Unknown cities",
                        f"{traffic.get('unknown_city_rate', 0):.1%}",
                        "no training history for these lanes",
                        ALERT if traffic.get("unknown_city_rate", 0) > 0.15 else INK,
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span("Drift status  ", style={"fontSize": "12px", "color": MUTED}),
                    badge(drift.get("status", "unknown")),
                ],
                style={"marginTop": "16px"},
            ),
        ],
        id="live-headline",
    )


def _live_series(frame) -> html.Div:
    """Predicted and confirmed rates side by side, refreshed on a timer.

    The actual line trails the predicted one because rates settle after they
    are quoted. That trailing gap is the delayed feedback problem made visible
    rather than described.

    Args:
        frame: Logged predictions with any outcomes joined.

    Returns:
        The chart panel.
    """
    if frame is None or frame.empty:
        return html.Div()

    return card(
        [
            section(
                "Predicted against actual",
                "Both lines are daily averages. The actual line stops short of the "
                "predicted one because those loads have not settled yet, and the "
                "shaded stretch is the traffic still waiting. Watch it fill in from "
                "the left while the replay runs.",
            ),
            dcc.Graph(
                id="live-predicted-actual",
                figure=predicted_vs_actual_series(frame),
                config={"displayModeBar": False},
                animate=False,
            ),
            dcc.Graph(
                id="live-error",
                figure=error_over_time(frame),
                config={"displayModeBar": False},
                animate=False,
            ),
        ]
    )


def _volume_chart(daily) -> html.Div:
    """Predictions per day, split by whether they can be scored.

    Args:
        daily: Per day metrics.

    Returns:
        The chart panel.
    """
    if daily is None or daily.empty:
        return html.Div()

    figure = go.Figure()
    figure.add_bar(
        x=daily["day"],
        y=daily["n_predictions"],
        name="predicted",
        marker_color=TEAL,
        opacity=0.35,
    )
    figure.add_bar(
        x=daily["day"],
        y=daily["n_scored"],
        name="outcome received",
        marker_color=TEAL,
    )
    figure.update_layout(
        **CHART_LAYOUT,
        barmode="overlay",
        height=280,
        title="Volume, and how much of it can be scored",
    )

    return card(
        [
            section(
                "Traffic",
                "The gap between the two bars is traffic still waiting on a "
                "confirmed rate. Freight settles days after it is quoted, so a gap "
                "at the right hand edge is normal.",
            ),
            dcc.Graph(id="live-volume", figure=figure, config={"displayModeBar": False}),
        ]
    )


def _accuracy_chart(daily) -> html.Div:
    """Error over time against the holdout figure.

    Args:
        daily: Per day metrics.

    Returns:
        The chart panel.
    """
    if daily is None or daily.empty or "mape" not in daily.columns:
        return html.Div()

    scored = daily[daily["mape"].notna()]
    if scored.empty:
        return html.Div()

    figure = go.Figure()
    figure.add_scatter(
        x=scored["day"],
        y=scored["mape"],
        mode="lines+markers",
        name="live MAPE",
        line={"color": TEAL, "width": 2.5},
        marker={"size": 5},
    )
    figure.add_hline(
        y=1.88,
        line_dash="dash",
        line_color=MUTED,
        annotation_text="holdout 1.88%",
        annotation_position="right",
    )
    figure.update_layout(
        **CHART_LAYOUT,
        height=280,
        title="Live error against the holdout figure",
        yaxis_title="MAPE (%)",
    )

    return card(
        [
            section(
                "Accuracy over time",
                "The dashed line is what the model scored during validation. Live "
                "error drifting above it means traffic has moved away from what "
                "the model learned.",
            ),
            dcc.Graph(id="live-accuracy", figure=figure, config={"displayModeBar": False}),
        ]
    )


def _signals(traffic: dict, drift: dict) -> html.Div:
    """Anything worth acting on, in plain words.

    Args:
        traffic: Traffic summary.
        drift: Drift report.

    Returns:
        The panel, or nothing when all is well.
    """
    notes = list(drift.get("notes", []))

    if drift.get("date_beyond_training_rate", 0) > 0.5:
        notes.append(
            f"{drift['date_beyond_training_rate']:.0%} of traffic is dated after the "
            "end of the training data, so the seasonal component is extrapolating."
        )

    if drift.get("drifted_features"):
        notes.append("Features that have moved: " + ", ".join(drift["drifted_features"]))

    if not notes:
        return card(
            [
                section("Signals"),
                html.Div(
                    "Nothing to act on. Traffic matches the training data.",
                    style={"fontSize": "13px", "color": OK},
                ),
            ]
        )

    return card(
        [
            section("Signals", "Things worth knowing about current traffic."),
            html.Ul(
                [
                    html.Li(note, style={"marginBottom": "8px", "lineHeight": "1.55"})
                    for note in notes
                ],
                style={"fontSize": "13px", "color": INK, "paddingLeft": "20px", "margin": 0},
            ),
        ]
    )
