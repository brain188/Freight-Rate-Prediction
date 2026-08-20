"""The performance tab: where the model is doing well and where it is not.

An overall error figure hides the group the model handles badly. Splitting by
equipment and distance is how a problem confined to short hauls or one trailer
type becomes visible instead of being averaged away.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html

from dashboard.theme import (
    ALERT,
    CHART_LAYOUT,
    MUTED,
    OK,
    TABLE_CELL_STYLE,
    TABLE_CONTAINER_STYLE,
    TABLE_DATA_STYLE,
    TABLE_HEADER_STYLE,
    TABLE_ROW_HOVER,
    TEAL,
    card,
    empty_state,
    metric,
    row,
    section,
)

# What the model scored during validation, for comparison against live traffic.
HOLDOUT = {"rmse": 67.65, "mae": 45.11, "mape": 1.88, "r2": 0.9976, "bias": 0.10}


def render(performance: dict, by_equipment, by_distance, daily) -> html.Div:
    """Build the performance tab.

    Args:
        performance: Snapshot from the performance endpoint.
        by_equipment: Metrics grouped by trailer type.
        by_distance: Metrics grouped by distance band.
        daily: Per day metrics.

    Returns:
        The tab content.
    """
    if not performance.get("n_scored"):
        return card(empty_state(
            "No outcomes reported yet, so nothing can be scored.",
            "Freight settles after it is quoted. Report actual rates to "
            "POST /actuals, or run the replay simulator.",
        ))

    return html.Div([
        _comparison(performance),
        _segments(by_equipment, by_distance),
        _feedback(performance, daily),
    ])


def _comparison(performance: dict) -> html.Div:
    """Live metrics next to what the model scored during validation.

    Args:
        performance: Performance snapshot.

    Returns:
        The comparison panel.
    """
    live = performance.get("metrics", {})

    rows = []
    for name, holdout_value in HOLDOUT.items():
        live_value = live.get(name)
        if live_value is None:
            continue

        if name in ("rmse", "mae", "bias"):
            live_text, holdout_text = f"${live_value:,.2f}", f"${holdout_value:,.2f}"
        elif name == "mape":
            live_text, holdout_text = f"{live_value:.2f}%", f"{holdout_value:.2f}%"
        else:
            live_text, holdout_text = f"{live_value:.4f}", f"{holdout_value:.4f}"

        # Bias is signed, so its size matters rather than its direction.
        if name == "bias":
            worse = abs(live_value) > abs(holdout_value) * 2
        elif name == "r2":
            worse = live_value < holdout_value - 0.005
        else:
            worse = live_value > holdout_value * 1.3

        rows.append({
            "metric": name.upper(),
            "live": live_text,
            "holdout": holdout_text,
            "flag": "worse" if worse else "",
        })

    return card([
        section(
            "Live against validation",
            f"Live figures come from {performance['n_scored']:,} predictions with a "
            f"confirmed rate, which is {performance.get('coverage', 0):.0%} of traffic "
            f"in the window. Holdout figures are from September and October, "
            f"months the model never trained on.",
        ),
        dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "Metric", "id": "metric"},
                {"name": "Live traffic", "id": "live"},
                {"name": "Validation holdout", "id": "holdout"},
                {"name": "", "id": "flag"},
            ],
            style_table=TABLE_CONTAINER_STYLE,
            style_cell={**TABLE_CELL_STYLE, "textAlign": "right"},
            style_cell_conditional=[{"if": {"column_id": "metric"}, "textAlign": "left"}],
            style_header=TABLE_HEADER_STYLE,
            style_data=TABLE_DATA_STYLE,
            css=TABLE_ROW_HOVER,
            style_data_conditional=[{
                "if": {"filter_query": "{flag} = 'worse'"},
                "color": ALERT, "fontWeight": 600,
            }],
        ),
    ])


def _segment_chart(frame, title: str, label: str) -> html.Div:
    """One bar chart of error by group.

    Args:
        frame: Segment metrics.
        title: Chart title.
        label: Axis label for the grouping.

    Returns:
        The chart, or nothing when there is no data.
    """
    if frame is None or frame.empty:
        return html.Div()

    scored = frame[frame["mape"].notna()]
    if scored.empty:
        return html.Div()

    colours = [ALERT if value > HOLDOUT["mape"] * 2 else TEAL for value in scored["mape"]]

    figure = go.Figure()
    figure.add_bar(
        x=scored["group"], y=scored["mape"], marker_color=colours,
        text=[f"{v:.2f}%" for v in scored["mape"]], textposition="outside",
        customdata=scored["n_scored"],
        hovertemplate="%{x}<br>MAPE %{y:.2f}%<br>%{customdata} scored<extra></extra>",
    )
    figure.add_hline(
        y=HOLDOUT["mape"], line_dash="dash", line_color=MUTED,
        annotation_text="holdout", annotation_position="right",
    )
    figure.update_layout(
        **CHART_LAYOUT, height=300, title=title,
        xaxis_title=label, yaxis_title="MAPE (%)", showlegend=False,
    )

    return html.Div(
        dcc.Graph(figure=figure, config={"displayModeBar": False}),
        style={"flex": "1", "minWidth": "340px"},
    )


def _segments(by_equipment, by_distance) -> html.Div:
    """Error broken down two ways.

    Args:
        by_equipment: Metrics by trailer type.
        by_distance: Metrics by distance band.

    Returns:
        The panel.
    """
    return card([
        section(
            "Where the error sits",
            "Short hauls were the hardest band during validation, at roughly four "
            "times the error of the middle of the range. Bars above the dashed "
            "line are doing worse than the model did on unseen months.",
        ),
        row([
            _segment_chart(by_equipment, "By trailer type", "equipment"),
            _segment_chart(by_distance, "By distance band", "miles"),
        ]),
    ])


def _feedback(performance: dict, daily) -> html.Div:
    """How quickly outcomes are arriving.

    Args:
        performance: Performance snapshot.
        daily: Per day metrics.

    Returns:
        The panel.
    """
    delay = performance.get("median_feedback_days")

    children = [
        section(
            "Feedback",
            "A rate is quoted now and confirmed later, so accuracy can only ever "
            "be measured on the traffic that has come back. If this delay grows, "
            "the model is being judged on increasingly stale evidence.",
        ),
        row([
            metric(
                "Median delay",
                f"{delay:.1f} days" if delay is not None else "n/a",
                "quote to confirmation",
            ),
            metric(
                "Scored",
                f"{performance.get('n_scored', 0):,}",
                f"of {performance.get('n_predictions', 0):,} predictions",
            ),
            metric(
                "Reliable",
                "yes" if performance.get("is_reliable") else "not yet",
                "enough outcomes to trust the figures",
                OK if performance.get("is_reliable") else MUTED,
            ),
        ]),
    ]

    if daily is not None and not daily.empty:
        figure = go.Figure()
        figure.add_scatter(
            x=daily["day"],
            y=(daily["n_scored"] / daily["n_predictions"] * 100).round(1),
            mode="lines+markers", line={"color": TEAL, "width": 2},
            marker={"size": 4}, name="coverage",
        )
        figure.update_layout(
            **CHART_LAYOUT, height=240,
            title="Share of each day's traffic that has an outcome",
            yaxis_title="coverage (%)",
        )
        children.append(dcc.Graph(figure=figure, config={"displayModeBar": False}))

    return card(children)