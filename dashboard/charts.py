"""Chart builders shared between tabs.

Two things here are worth explaining. The predicted against actual series is
built to be refreshed on a timer, so it fills in from the left as a replay
runs. And the density plots split production traffic into the part that still
resembles training and the part that does not, because a single production
curve hides which half of the traffic is causing a drift signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import ALERT, CHART_LAYOUT, MUTED, TEAL, WARN
from src.logger import get_logger

logger = get_logger(__name__)

# Colours for the three series on every density plot, so the legend means the
# same thing wherever it appears.
REFERENCE_COLOUR = "#7A8B90"
PRODUCTION_OK_COLOUR = TEAL
PRODUCTION_DRIFT_COLOUR = ALERT

LEGEND_REFERENCE = "Training data"
LEGEND_OK = "Production data - ok"
LEGEND_DRIFT = "Production data - drift"

# Points along the x axis for a smooth density curve.
DENSITY_POINTS = 220

# Below this many rows a kernel density estimate is more noise than signal.
MIN_DENSITY_ROWS = 20


def _rgba(hex_colour: str, alpha: float) -> str:
    """Turn a hex colour into rgba, for shaded fills.

    Args:
        hex_colour: Colour as #RRGGBB.
        alpha: Opacity between 0 and 1.

    Returns:
        An rgba string.
    """
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _density(values: pd.Series, grid: np.ndarray) -> np.ndarray | None:
    """Estimate a smooth density over a fixed grid.

    A shared grid is used so the three curves can be drawn on one axis and
    compared by eye rather than by reading numbers off separate charts.

    Args:
        values: The observations.
        grid: X positions to evaluate the density at.

    Returns:
        Density values, or None when there is too little data.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean) < MIN_DENSITY_ROWS or clean.nunique() < 3:
        return None

    try:
        from scipy.stats import gaussian_kde

        return gaussian_kde(clean.to_numpy())(grid)
    
    except ImportError:
        logger.warning(
            "SciPy is unavailable; falling back to histogram density."
        )

    except Exception as exc:
        logger.warning(
            "KDE density estimation failed; falling back to histogram: %s",
            exc,
            exc_info=True,
        )
        # Falls back to a histogram, which is coarser but always works.
        counts, edges = np.histogram(clean, bins=40, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        return np.interp(grid, centres, counts, left=0.0, right=0.0)


def _shared_grid(*series: pd.Series) -> np.ndarray | None:
    """Build one x axis covering every series passed in.

    Args:
        *series: The distributions to be plotted together.

    Returns:
        The grid, or None when nothing usable was supplied.
    """
    pooled = pd.concat(
        [pd.to_numeric(s, errors="coerce").dropna() for s in series if s is not None]
    )

    if pooled.empty:
        return None

    # Trimmed at the extremes so one outlier does not flatten the whole curve.
    low, high = pooled.quantile(0.005), pooled.quantile(0.995)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    padding = (high - low) * 0.08
    return np.linspace(low - padding, high + padding, DENSITY_POINTS)


def density_plot(
    reference: pd.Series,
    production_ok: pd.Series,
    production_drift: pd.Series,
    title: str,
    x_label: str,
) -> go.Figure:
    """Compare training against production, split by drift status.

    Three curves rather than two. Splitting production means a reader can see
    whether a shift comes from the whole of production or only from the part
    already flagged as unfamiliar, which points straight at the cause.

    Args:
        reference: Values from the training data.
        production_ok: Production values on lanes the model has history for.
        production_drift: Production values flagged as unfamiliar.
        title: Chart title.
        x_label: Axis label.

    Returns:
        The figure.
    """
    figure = go.Figure()
    grid = _shared_grid(reference, production_ok, production_drift)

    if grid is None:
        figure.update_layout(**CHART_LAYOUT, height=320, title=title)
        figure.add_annotation(
            text="Not enough data to plot", showarrow=False,
            font={"size": 13, "color": MUTED},
        )
        return figure

    series = [
        (reference, LEGEND_REFERENCE, REFERENCE_COLOUR, "solid", 0.22),
        (production_ok, LEGEND_OK, PRODUCTION_OK_COLOUR, "solid", 0.16),
        (production_drift, LEGEND_DRIFT, PRODUCTION_DRIFT_COLOUR, "dot", 0.16),
    ]

    for values, label, colour, dash, alpha in series:
        if values is None:
            continue

        density = _density(values, grid)
        if density is None:
            continue

        figure.add_scatter(
            x=grid, y=density, mode="lines", name=f"{label}  (n={len(values.dropna()):,})",
            line={"color": colour, "width": 2.4, "dash": dash},
            fill="tozeroy", fillcolor=_rgba(colour, alpha),
            hovertemplate=f"{label}<br>%{{x:,.0f}}<extra></extra>",
        )

    # Copied and overridden rather than passed twice, since CHART_LAYOUT
    # already sets hovermode and legend.
    layout = dict(CHART_LAYOUT)
    layout["hovermode"] = "closest"
    layout["legend"] = {"orientation": "h", "y": -0.22, "font": {"size": 11}}

    figure.update_layout(
        **layout, height=320, title=title,
        xaxis_title=x_label, yaxis_title="density",
    )
    figure.update_yaxes(showticklabels=False)

    return figure


def _time_axis(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Choose which date to plot along the x axis.

    A replay compresses months of traffic into minutes of wall clock, so
    grouping by when a prediction was made would collapse everything onto one
    point. Grouping by the date the load moves spreads it out the way it would
    look in production, where the two are close together anyway.

    Args:
        frame: Logged predictions.

    Returns:
        The dates to group by, and a label for them.
    """
    served = pd.to_datetime(frame["predicted_at"])

    if served.dt.floor("D").nunique() > 3 or "load_date" not in frame.columns:
        return served, "served"

    return pd.to_datetime(frame["load_date"]), "load date"


def predicted_vs_actual_series(frame: pd.DataFrame, resample: str = "D") -> go.Figure:
    """Track predicted and confirmed rates side by side over time.

    Built to be refreshed on a timer. As a replay runs, the predicted line
    extends to the right while the actual line trails it, because rates settle
    after they are quoted. That lag is the shape of the problem.

    Args:
        frame: Predictions with any outcomes joined.
        resample: Pandas offset to average over.

    Returns:
        The figure.
    """
    figure = go.Figure()

    if frame is None or frame.empty:
        figure.update_layout(**CHART_LAYOUT, height=330, title="Predicted against actual")
        figure.add_annotation(
            text="No traffic yet", showarrow=False,
            font={"size": 13, "color": MUTED},
        )
        return figure

    working = frame.copy()
    axis, axis_label = _time_axis(working)
    working["bucket"] = axis.dt.floor(resample)

    predicted = working.groupby("bucket")["predicted_rate"].mean()
    scored = working[working["actual_rate"].notna()]
    actual = scored.groupby("bucket")["actual_rate"].mean() if not scored.empty else None

    figure.add_scatter(
        x=predicted.index, y=predicted.to_numpy(), mode="lines+markers",
        name="Predicted", line={"color": TEAL, "width": 2.6}, marker={"size": 5},
        hovertemplate="Predicted $%{y:,.0f}<extra></extra>",
    )

    if actual is not None and not actual.empty:
        figure.add_scatter(
            x=actual.index, y=actual.to_numpy(), mode="lines+markers",
            name="Actual", line={"color": WARN, "width": 2.6, "dash": "dot"},
            marker={"size": 5},
            hovertemplate="Actual $%{y:,.0f}<extra></extra>",
        )

        # Shade the stretch that has no outcomes yet, so the gap reads as
        # pending rather than as the model having stopped.
        last_actual = actual.index.max()
        if predicted.index.max() > last_actual:
            figure.add_vrect(
                x0=last_actual, x1=predicted.index.max(),
                fillcolor=_rgba(MUTED, 0.07), line_width=0,
                annotation_text="awaiting outcomes",
                annotation_position="top left",
                annotation={"font": {"size": 10, "color": MUTED}},
            )

    figure.update_layout(
        **CHART_LAYOUT, height=330,
        title=f"Predicted against actual, daily average by {axis_label}",
        yaxis_title="rate ($)",
    )
    return figure


def error_over_time(frame: pd.DataFrame, resample: str = "D") -> go.Figure:
    """Show the gap between predicted and actual as it develops.

    Args:
        frame: Predictions with any outcomes joined.
        resample: Pandas offset to average over.

    Returns:
        The figure.
    """
    figure = go.Figure()
    scored = frame[frame["actual_rate"].notna()] if frame is not None else None

    if scored is None or scored.empty:
        figure.update_layout(**CHART_LAYOUT, height=260, title="Prediction error over time")
        figure.add_annotation(
            text="No confirmed rates yet", showarrow=False,
            font={"size": 13, "color": MUTED},
        )
        return figure

    working = scored.copy()
    axis, _ = _time_axis(working)
    working["bucket"] = axis.dt.floor(resample)
    working["error"] = working["actual_rate"] - working["predicted_rate"]

    grouped = working.groupby("bucket")["error"]
    mean_error = grouped.mean()
    spread = grouped.std().fillna(0.0)

    figure.add_scatter(
        x=mean_error.index, y=(mean_error + spread).to_numpy(),
        mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip",
    )
    figure.add_scatter(
        x=mean_error.index, y=(mean_error - spread).to_numpy(),
        mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor=_rgba(TEAL, 0.13), name="one standard deviation",
        hoverinfo="skip",
    )
    figure.add_scatter(
        x=mean_error.index, y=mean_error.to_numpy(), mode="lines+markers",
        name="mean error", line={"color": TEAL, "width": 2.4}, marker={"size": 4},
        hovertemplate="$%{y:,.0f}<extra></extra>",
    )
    figure.add_hline(y=0, line_dash="dash", line_color=MUTED)

    figure.update_layout(
        **CHART_LAYOUT, height=260,
        title="Prediction error over time, actual minus predicted",
        yaxis_title="error ($)",
    )
    return figure


def split_production(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Split production traffic into the familiar part and the drifted part.

    A load counts as drifted when the model had no history for one of its
    cities. That is the flag the model itself raised at prediction time, so
    the split reflects the model's own uncertainty rather than an arbitrary
    threshold applied afterwards.

    Args:
        frame: Logged predictions.
        column: Which column to return.

    Returns:
        The values on familiar lanes, and the values on unfamiliar ones.
    """
    if frame is None or frame.empty or column not in frame.columns:
        empty = pd.Series(dtype=float)
        return empty, empty

    drifted = frame["unknown_city"].fillna(False).astype(bool)
    return frame.loc[~drifted, column], frame.loc[drifted, column]