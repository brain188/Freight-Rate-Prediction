"""The registry tab: what has been trained and what is being served.

Shows the serving model, every registered version and the recent training runs.
Each row links out to MLflow rather than trying to reproduce its UI, because
MLflow already does run comparison and artifact browsing well and duplicating
that here would be worse in every respect.
"""

from __future__ import annotations

from dash import dash_table, html

from dashboard.theme import (
    ALERT,
    INK,
    LINE,
    MONO,
    MUTED,
    OK,
    TEAL,
    TEAL_LIGHT,
    card,
    empty_state,
    metric,
    row,
    section,
)

# Metrics worth showing per run, in the order they mean most.
RUN_METRICS = [
    ("holdout_rmse", "Holdout RMSE", "${:,.2f}"),
    ("holdout_mape", "Holdout MAPE", "{:.2f}%"),
    ("cv_rmse", "CV RMSE", "${:,.2f}"),
    ("improvement_over_baseline_pct", "vs baseline", "{:.1f}%"),
]


def render(model_info: dict, runs: list, versions: list, tracker) -> html.Div:
    """Build the registry tab.

    Args:
        model_info: Metadata for the model currently being served.
        runs: Recent training runs from MLflow.
        versions: Registered model versions.
        tracker: The MLflow tracker, used to build deep links.

    Returns:
        The tab content.
    """
    return html.Div([
        _serving_model(model_info, tracker),
        _versions(versions, tracker),
        _runs(runs, tracker),
    ])


def _link(url: str | None, text: str) -> html.A | html.Span:
    """Build an outbound link, or plain text when there is nowhere to go.

    Args:
        url: The destination, or None.
        text: What to show.

    Returns:
        A link or a span.
    """
    if not url:
        return html.Span(text, style={"color": MUTED})

    return html.A(
        text,
        href=url,
        target="_blank",
        style={"color": TEAL, "fontWeight": 600, "textDecoration": "none"},
    )


def _serving_model(model_info: dict, tracker) -> html.Div:
    """What the API is serving right now.

    Args:
        model_info: Metadata for the serving model.
        tracker: The MLflow tracker.

    Returns:
        The panel.
    """
    if not model_info:
        return card(empty_state(
            "Could not read the serving model.",
            "Check that the API is running and has a model loaded.",
        ))

    validation = model_info.get("validation", {})
    holdout = validation.get("holdout", {})

    children = [
        section(
            "Currently serving",
            "Read from the API itself, so this is the model actually answering "
            "requests rather than the newest one in the registry.",
        ),
        row([
            metric("Trained", model_info.get("trained_at", "unknown")[:16].replace("T", " ")),
            metric("Estimator", model_info.get("estimator", "unknown")),
            metric("Features", str(model_info.get("n_features", 0))),
            metric("Known cities", str(model_info.get("known_cities", 0))),
        ]),
    ]

    if holdout:
        children.append(html.Div(style={"height": "16px"}))
        children.append(row([
            metric("Holdout RMSE", f"${holdout.get('rmse', 0):,.2f}"),
            metric("Holdout MAPE", f"{holdout.get('mape', 0):.2f}%"),
            metric("R squared", f"{holdout.get('r2', 0):.4f}"),
            metric(
                "vs baseline",
                f"{validation.get('improvement_over_baseline_pct', 0):.1f}%",
                "better than median rate per mile",
                OK,
            ),
        ]))

    children.append(html.Div([
        html.Span("Trained on ", style={"fontSize": "12px", "color": MUTED}),
        html.Span(
            f"{model_info.get('training_rows', 0):,} loads from "
            f"{model_info.get('training_from', '')} to {model_info.get('training_to', '')}",
            style={"fontSize": "12px", "color": INK, "fontFamily": MONO},
        ),
    ], style={"marginTop": "18px"}))

    return card(children)


def _versions(versions: list, tracker) -> html.Div:
    """Every registered model version.

    Args:
        versions: Version summaries.
        tracker: The MLflow tracker.

    Returns:
        The panel.
    """
    if not versions:
        return card([
            section(
                "Model registry",
                "Nothing registered yet. Training runs are registered "
                "automatically once a tracking server is reachable.",
            ),
            _mlflow_hint(tracker),
        ])

    rows = [{
        "version": f"v{v.version}",
        "stage": v.stage,
        "created": v.created_at,
        "run": v.run_id[:8],
    } for v in versions]

    return card([
        section(
            "Model registry",
            "Every version that has been registered. The production stage is the "
            "one a deployment should be pulling.",
        ),
        dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "Version", "id": "version"},
                {"name": "Stage", "id": "stage"},
                {"name": "Created", "id": "created"},
                {"name": "Run", "id": "run"},
            ],
            style_cell={
                "fontFamily": MONO, "fontSize": "13px",
                "padding": "10px 14px", "textAlign": "left", "border": "none",
            },
            style_header={
                "backgroundColor": TEAL_LIGHT, "fontWeight": 700,
                "color": INK, "border": "none", "fontSize": "12px",
            },
            style_data={"borderBottom": f"1px solid {LINE}"},
            style_data_conditional=[{
                "if": {"filter_query": "{stage} = 'Production'"},
                "color": OK, "fontWeight": 600,
            }],
        ),
        html.Div([
            _link(
                tracker.model_url(versions[0].name, versions[0].version),
                "Open the registry in MLflow →",
            )
        ], style={"marginTop": "14px", "fontSize": "13px"}),
    ])


def _runs(runs: list, tracker) -> html.Div:
    """Recent training runs, each linking to MLflow.

    Args:
        runs: Run summaries.
        tracker: The MLflow tracker.

    Returns:
        The panel.
    """
    if not runs:
        return card([
            section(
                "Training runs",
                "No runs recorded. Every call to entrypoint/train.py logs its "
                "parameters, metrics and artifacts once MLflow is reachable.",
            ),
            _mlflow_hint(tracker),
        ])

    rows = []
    for run in runs:
        entry = {
            "run": run.short_id,
            "name": run.run_name,
            "started": run.started_at,
            "status": run.status,
        }
        for key, label, template in RUN_METRICS:
            value = run.metrics.get(key)
            entry[key] = template.format(value) if value is not None else "—"
        rows.append(entry)

    columns = [
        {"name": "Run", "id": "run"},
        {"name": "Name", "id": "name"},
        {"name": "Started", "id": "started"},
    ] + [{"name": label, "id": key} for key, label, _ in RUN_METRICS]

    return card([
        section(
            "Training runs",
            "Every run with its holdout and cross validation scores. Click through "
            "for parameters, artifacts and side by side comparison in MLflow, "
            "which does that far better than reproducing it here would.",
        ),
        dash_table.DataTable(
            data=rows,
            columns=columns,
            style_cell={
                "fontFamily": MONO, "fontSize": "12.5px",
                "padding": "9px 13px", "textAlign": "right", "border": "none",
            },
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ("run", "name", "started")
            ],
            style_header={
                "backgroundColor": TEAL_LIGHT, "fontWeight": 700,
                "color": INK, "border": "none", "fontSize": "12px",
            },
            style_data={"borderBottom": f"1px solid {LINE}"},
            page_size=12,
        ),
        html.Div([
            html.Span("Latest run: ", style={"fontSize": "13px", "color": MUTED}),
            _link(tracker.run_url(runs[0].run_id), f"open {runs[0].short_id} in MLflow →"),
        ], style={"marginTop": "14px"}),
    ])


def _mlflow_hint(tracker) -> html.Div:
    """Explain how to get MLflow running.

    Args:
        tracker: The MLflow tracker.

    Returns:
        The hint block.
    """
    reachable = tracker.is_available

    if reachable and not tracker.ui_url:
        message = (
            "MLflow is writing to a local directory, which has no web interface. "
            "Point MLFLOW_TRACKING_URI at a server to enable the links above."
        )
    elif reachable:
        message = f"MLflow is reachable at {tracker.ui_url}."
    else:
        message = "No tracking server reachable. Start one with the command below."

    return html.Div([
        html.Div(message, style={"fontSize": "13px", "color": INK, "marginBottom": "10px"}),
        html.Div(
            "mlflow server --host 0.0.0.0 --port 5000",
            style={
                "fontFamily": MONO, "fontSize": "12px", "color": INK,
                "background": TEAL_LIGHT, "padding": "10px 14px", "borderRadius": "6px",
            },
        ),
    ], style={"marginTop": "8px"})