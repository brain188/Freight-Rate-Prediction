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

FONT = "'Inter', -apple-system, 'Segoe UI', Helvetica, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

RADIUS = "14px"
SHADOW_SM = "0 1px 2px rgba(14, 42, 49, 0.05), 0 1px 3px rgba(14, 42, 49, 0.04)"
SHADOW_MD = "0 2px 10px rgba(14, 42, 49, 0.06), 0 12px 28px rgba(14, 42, 49, 0.07)"

CHART_LAYOUT = {
    "font": {"family": FONT, "size": 12, "color": INK},
    "paper_bgcolor": PANEL,
    "plot_bgcolor": PANEL,
    "margin": {"l": 55, "r": 25, "t": 50, "b": 45},
    "hovermode": "x unified",
    "hoverlabel": {
        "bgcolor": PANEL, "bordercolor": LINE,
        "font": {"family": FONT, "size": 12, "color": INK},
    },
    "xaxis": {"gridcolor": "#EBF1F2", "zeroline": False, "linecolor": LINE},
    "yaxis": {"gridcolor": "#EBF1F2", "zeroline": False, "linecolor": LINE},
    "legend": {
        "orientation": "h", "y": -0.2,
        "font": {"size": 11, "color": MUTED},
    },
    "title_font": {"size": 13.5, "color": INK, "family": FONT},
    "title_x": 0.0,
    "title_xanchor": "left",
}

# Shared look for every dash_table.DataTable, so a reader learns one table
# style once rather than re-reading it on every tab.
TABLE_CELL_STYLE = {
    "fontFamily": MONO, "fontSize": "12.5px", "padding": "11px 14px",
    "border": "none", "color": INK,
}
TABLE_HEADER_STYLE = {
    "backgroundColor": BACKGROUND, "fontWeight": 700, "color": MUTED,
    "border": "none", "fontSize": "10.5px", "textTransform": "uppercase",
    "letterSpacing": "0.6px", "borderBottom": f"2px solid {LINE}",
    "fontFamily": FONT,
}
TABLE_DATA_STYLE = {"borderBottom": f"1px solid {LINE}"}
TABLE_CONTAINER_STYLE = {
    "borderRadius": "10px", "overflow": "hidden", "border": f"1px solid {LINE}",
}
# Row hover, scoped to whichever DataTable's `css` prop it is passed to.
TABLE_ROW_HOVER = [
    {"selector": "tr:hover td", "rule": f"background-color: {TEAL_LIGHT} !important;"},
]


def _rgba(hex_colour: str, alpha: float) -> str:
    """Turn a hex colour into rgba, for tints and soft fills.

    Args:
        hex_colour: Colour as #RRGGBB.
        alpha: Opacity between 0 and 1.

    Returns:
        An rgba string.
    """
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def card(children, className: str = "", **style) -> html.Div:
    """Wrap content in a panel.

    Args:
        children: What to put inside.
        className: Extra CSS classes, appended to the hover-lift class.
        **style: Extra CSS to merge in, or an `id` to set on the panel.

    Returns:
        The panel.
    """
    base = {
        "background": PANEL,
        "border": f"1px solid {LINE}",
        "borderRadius": RADIUS,
        "padding": "22px 24px",
        "marginBottom": "18px",
        "boxShadow": SHADOW_SM,
    }
    node_id = style.pop("id", None)
    props = {"style": {**base, **style}, "className": f"dash-card {className}".strip()}
    if node_id:
        props["id"] = node_id
    return html.Div(children, **props)


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
            "fontSize": "10.5px", "color": MUTED, "textTransform": "uppercase",
            "letterSpacing": "0.7px", "fontWeight": 700, "marginBottom": "8px",
        }),
        html.Div(value, style={
            "fontSize": "30px", "fontWeight": 700, "color": colour, "lineHeight": "1.1",
            "fontFamily": MONO, "letterSpacing": "-0.02em",
            "fontVariantNumeric": "tabular-nums",
        }),
    ]

    if note:
        children.append(html.Div(note, style={
            "fontSize": "11.5px", "color": MUTED, "marginTop": "7px",
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
    return html.Span([
        html.Span(style={
            "display": "inline-block", "width": "6px", "height": "6px",
            "borderRadius": "50%", "background": colour, "marginRight": "7px",
        }),
        html.Span(STATUS_LABELS.get(status, status)),
    ], style={
        "display": "inline-flex", "alignItems": "center",
        "background": _rgba(colour, 0.11), "color": colour,
        "border": f"1px solid {_rgba(colour, 0.28)}",
        "padding": "4px 12px 4px 9px", "borderRadius": "20px",
        "fontSize": "11px", "fontWeight": 700, "letterSpacing": "0.3px",
    })


def section(title: str, subtitle: str = "") -> html.Div:
    """A heading with optional explanatory line.

    Args:
        title: The heading.
        subtitle: A line explaining what follows.

    Returns:
        The heading block.
    """
    children = [html.Div(title, style={
        "fontSize": "15.5px", "fontWeight": 700, "color": INK,
        "marginBottom": "4px", "letterSpacing": "-0.01em",
    })]

    if subtitle:
        children.append(html.Div(subtitle, style={
            "fontSize": "12.5px", "color": MUTED, "lineHeight": "1.55",
            "maxWidth": "760px",
        }))

    return html.Div(children, style={"marginBottom": "16px"})


def empty_state(message: str, hint: str = "") -> html.Div:
    """Shown when there is nothing to display.

    Args:
        message: What is missing.
        hint: How to fix it.

    Returns:
        The placeholder.
    """
    children = [
        html.Div("◇", style={"fontSize": "20px", "color": "#B7C6C9", "marginBottom": "10px"}),
        html.Div(message, style={"fontSize": "14px", "color": MUTED, "fontWeight": 500}),
    ]

    if hint:
        children.append(html.Div(hint, style={
            "fontSize": "12px", "color": MUTED, "marginTop": "10px", "fontFamily": MONO,
            "background": PANEL, "display": "inline-block", "padding": "6px 12px",
            "borderRadius": "6px", "border": f"1px solid {LINE}",
        }))

    return html.Div(children, style={
        "padding": "52px 20px", "textAlign": "center",
        "background": BACKGROUND, "borderRadius": "10px",
        "border": f"1px dashed {LINE}",
    })


def row(children, gap: str = "18px") -> html.Div:
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
