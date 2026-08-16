"""The drift tab: has traffic moved away from what the model learned.

Drift matters more here than in most projects. The model trained on January to
October and every load it prices afterwards is a forecast, so traffic drifting
away from the training data is the earliest warning available. It also arrives
before accuracy does, because it needs no confirmed rates.

Two views. A fast summary computed on every refresh, and the full Evidently
report loaded on demand because it is slow enough to be worth a button.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dash_table, dcc, html
from monitoring.drift import PSI_ALERT, PSI_WARN, UNKNOWN_CITY_ALERT, UNKNOWN_CITY_WARN

from dashboard.theme import (
    ALERT,
    CHART_LAYOUT,
    INK,
    LINE,
    MONO,
    MUTED,
    OK,
    TEAL,
    TEAL_LIGHT,
    WARN,
    badge,
    card,
    empty_state,
    metric,
    row,
    section,
)


def render(drift: dict, daily_unknown=None) -> html.Div:
    """Build the drift tab.

    Args:
        drift: The drift report.
        daily_unknown: Unknown city rate per day, if available.

    Returns:
        The tab content.
    """
    if not drift.get("n_current"):
        return card(empty_state(
            "No traffic to compare against the training data.",
            "python simulator/replay.py --speed 200",
        ))

    return html.Div([
        _summary(drift),
        _unknown_cities(drift, daily_unknown),
        _feature_table(drift),
        _evidently_panel(),
    ])


def _summary(drift: dict) -> html.Div:
    """The headline drift numbers.

    Args:
        drift: The drift report.

    Returns:
        The summary panel.
    """
    unknown = drift.get("unknown_city_rate", 0)
    unknown_colour = (
        ALERT if unknown >= UNKNOWN_CITY_ALERT
        else WARN if unknown >= UNKNOWN_CITY_WARN
        else OK
    )

    return card([
        html.Div([
            html.Span("Drift status  ", style={"fontSize": "13px", "color": MUTED}),
            badge(drift.get("status", "unknown")),
        ], style={"marginBottom": "18px"}),
        row([
            metric(
                "Unknown cities",
                f"{unknown:.1%}",
                "traffic with no training history",
                unknown_colour,
            ),
            metric(
                "Beyond training dates",
                f"{drift.get('date_beyond_training_rate', 0):.0%}",
                "loads dated after 31 October",
            ),
            metric(
                "Features drifted",
                str(len(drift.get("drifted_features", []))),
                "of those compared",
                ALERT if drift.get("drifted_features") else OK,
            ),
            metric(
                "Sample",
                f"{drift['n_current']:,}",
                f"against {drift['n_reference']:,} training rows",
            ),
        ]),
    ])


def _unknown_cities(drift: dict, daily_unknown) -> html.Div:
    """The unknown city signal, explained.

    Args:
        drift: The drift report.
        daily_unknown: Rate per day, if available.

    Returns:
        The panel.
    """
    rate = drift.get("unknown_city_rate", 0)

    children = [
        section(
            "Cities the model has never priced",
            "This is the most useful drift signal in the system and the cheapest. "
            "It comes from a flag raised at prediction time, needs no reference "
            "dataset, and is available immediately rather than waiting for rates "
            "to settle. Loads on these lanes are priced from position alone, with "
            "no local history behind them.",
        ),
    ]

    if daily_unknown is not None and not daily_unknown.empty:
        figure = go.Figure()
        figure.add_scatter(
            x=daily_unknown["day"], y=daily_unknown["unknown_city_rate"] * 100,
            mode="lines+markers", line={"color": TEAL, "width": 2.5},
            marker={"size": 5}, name="unknown city rate",
        )
        figure.add_hline(
            y=UNKNOWN_CITY_WARN * 100, line_dash="dot", line_color=WARN,
            annotation_text="watch", annotation_position="right",
        )
        figure.add_hline(
            y=UNKNOWN_CITY_ALERT * 100, line_dash="dot", line_color=ALERT,
            annotation_text="act", annotation_position="right",
        )
        figure.update_layout(
            **CHART_LAYOUT, height=290,
            title="Share of traffic on unfamiliar lanes",
            yaxis_title="%",
        )
        children.append(dcc.Graph(figure=figure, config={"displayModeBar": False}))

    if rate > 0:
        children.append(html.Div(
            f"At {rate:.1%}, roughly one load in "
            f"{max(round(1 / rate), 1)} involves a city absent from the training "
            "data. Adding those markets to the next training set is the fix.",
            style={
                "fontSize": "13px", "color": INK, "background": TEAL_LIGHT,
                "padding": "13px 16px", "borderRadius": "7px", "marginTop": "12px",
                "lineHeight": "1.55",
            },
        ))

    return card(children)


def _feature_table(drift: dict) -> html.Div:
    """Per feature drift scores.

    Args:
        drift: The drift report.

    Returns:
        The panel.
    """
    features = drift.get("features", [])

    if not features:
        return html.Div()

    rows = [{
        "feature": f["feature"],
        "psi": f"{f['psi']:.4f}",
        "reference": f"{f['reference_mean']:,.0f}" if f.get("reference_mean") else "—",
        "current": f"{f['current_mean']:,.0f}" if f.get("current_mean") else "—",
        "status": f["status"],
    } for f in features]

    return card([
        section(
            "Feature distributions",
            f"Population Stability Index compares live traffic against the training "
            f"data. Below {PSI_WARN} is stable, above {PSI_ALERT} is a real shift. "
            "PSI is used rather than a statistical test because it does not grow "
            "with sample size, so it does not call every tiny difference significant "
            "once the traffic gets large.",
        ),
        dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "Feature", "id": "feature"},
                {"name": "PSI", "id": "psi"},
                {"name": "Training mean", "id": "reference"},
                {"name": "Live mean", "id": "current"},
                {"name": "Status", "id": "status"},
            ],
            style_cell={
                "fontFamily": MONO, "fontSize": "13px",
                "padding": "10px 14px", "textAlign": "right", "border": "none",
            },
            style_cell_conditional=[{"if": {"column_id": "feature"}, "textAlign": "left"}],
            style_header={
                "backgroundColor": TEAL_LIGHT, "fontWeight": 700,
                "color": INK, "border": "none", "fontSize": "12px",
            },
            style_data={"borderBottom": f"1px solid {LINE}"},
            style_data_conditional=[
                {"if": {"filter_query": "{status} = 'alert'"}, "color": ALERT, "fontWeight": 600},
                {"if": {"filter_query": "{status} = 'warn'"}, "color": WARN},
            ],
        ),
    ])


def _evidently_panel() -> html.Div:
    """The button that loads the full Evidently report.

    Loaded on demand rather than on every refresh, because building it takes
    several seconds and most visits do not need that depth.

    Returns:
        The panel.
    """
    return card([
        section(
            "Full Evidently report",
            "The summary above is computed on every refresh. This builds the "
            "complete Evidently data drift report, with per feature distribution "
            "comparisons and statistical tests. It takes a few seconds.",
        ),
        html.Button(
            "Build the Evidently report",
            id="build-evidently",
            n_clicks=0,
            style={
                "background": TEAL, "color": "#FFFFFF", "border": "none",
                "padding": "10px 20px", "borderRadius": "7px", "fontSize": "13px",
                "fontWeight": 600, "cursor": "pointer",
            },
        ),
        dcc.Loading(
            html.Div(id="evidently-output", style={"marginTop": "18px"}),
            type="dot",
            color=TEAL,
        ),
    ])