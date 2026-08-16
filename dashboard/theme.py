"""Colours and layout helpers shared across the dashboard.

Kept in one place so a chart and a status badge agree on what "alert" looks
like, rather than each tab picking its own red.
"""

from __future__ import annotations

from dash import html

INK = "#0E2A31"
MUTED = "#5B7076"
LINE = "#D9E2E4"
PANEL = "#FFFFFF"
BACKGROUND = "#F4F7F8"

TEAL = "#064A56"
TEAL_LIGHT = "#E8F0F1"

OK = "#1F7A4D"
WARN = "#B57200"
ALERT = "#B23A32"
UNKNOWN = "#7A8B90"

STATUS_COLOURS = {"ok": OK, "warn": WARN, "alert": ALERT, "unknown": UNKNOWN}

STATUS_LABELS = {
    "ok": "Healthy",
    "warn": "Watch",
    "alert": "Action needed",
    "unknown": "Not enough data",
}

FONT = "-apple-system, 'Segoe UI', Inter, Helvetica, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

CHART_LAYOUT = {
    "font": {"family": FONT, "size": 12, "color": INK},
    "paper_bgcolor": PANEL,
    "plot_bgcolor": PANEL,
    "margin": {"l": 55, "r": 25, "t": 45, "b": 45},
    "hovermode": "x unified",
    "xaxis": {"gridcolor": LINE, "zeroline": False},
    "yaxis": {"gridcolor": LINE, "zeroline": False},
    "legend": {"orientation": "h", "y": -0.2},
}


def card(children, **style) -> html.Div:
    """Wrap content in a panel.

    Args:
        children: What to put inside.
        **style: Extra CSS to merge in.

    Returns:
        The panel.
    """
    base = {
        "background": PANEL,
        "border": f"1px solid {LINE}",
        "borderRadius": "10px",
        "padding": "18px 20px",
        "marginBottom": "16px",
    }
    return html.Div(children, style={**base, **style})


def metric(label: str, value: str, note: str = "", colour: str = INK) -> html.Div:
    """A single headline number with its label.

    Args:
        label: What the number measures.
        value: The number, already formatted.
        note: Optional context below the number.
        colour: Colour for the value.

    Returns:
        The metric block.
    """
    children = [
        html.Div(label, style={
            "fontSize": "11px", "color": MUTED, "textTransform": "uppercase",
            "letterSpacing": "0.6px", "fontWeight": 600, "marginBottom": "6px",
        }),
        html.Div(value, style={
            "fontSize": "26px", "fontWeight": 700, "color": colour, "lineHeight": "1.1",
        }),
    ]

    if note:
        children.append(html.Div(note, style={
            "fontSize": "11px", "color": MUTED, "marginTop": "5px",
        }))

    return html.Div(children, style={"flex": "1", "minWidth": "150px"})


def badge(status: str) -> html.Span:
    """A coloured status pill.

    Args:
        status: One of "ok", "warn", "alert" or "unknown".

    Returns:
        The badge.
    """
    colour = STATUS_COLOURS.get(status, UNKNOWN)
    return html.Span(
        STATUS_LABELS.get(status, status),
        style={
            "background": colour, "color": "#FFFFFF", "padding": "3px 11px",
            "borderRadius": "11px", "fontSize": "11px", "fontWeight": 700,
            "letterSpacing": "0.4px",
        },
    )


def section(title: str, subtitle: str = "") -> html.Div:
    """A heading with optional explanatory line.

    Args:
        title: The heading.
        subtitle: A line explaining what follows.

    Returns:
        The heading block.
    """
    children = [html.Div(title, style={
        "fontSize": "15px", "fontWeight": 700, "color": INK, "marginBottom": "3px",
    })]

    if subtitle:
        children.append(html.Div(subtitle, style={
            "fontSize": "12px", "color": MUTED, "lineHeight": "1.5",
        }))

    return html.Div(children, style={"marginBottom": "14px"})


def empty_state(message: str, hint: str = "") -> html.Div:
    """Shown when there is nothing to display.

    Args:
        message: What is missing.
        hint: How to fix it.

    Returns:
        The placeholder.
    """
    children = [html.Div(message, style={"fontSize": "14px", "color": MUTED})]

    if hint:
        children.append(html.Div(hint, style={
            "fontSize": "12px", "color": MUTED, "marginTop": "8px", "fontFamily": MONO,
        }))

    return html.Div(children, style={
        "padding": "44px 20px", "textAlign": "center",
        "background": BACKGROUND, "borderRadius": "8px",
    })


def row(children, gap: str = "16px") -> html.Div:
    """Lay children out horizontally, wrapping on narrow screens.

    Args:
        children: What to lay out.
        gap: Space between items.

    Returns:
        The row.
    """
    return html.Div(children, style={
        "display": "flex", "gap": gap, "flexWrap": "wrap", "alignItems": "stretch",
    })