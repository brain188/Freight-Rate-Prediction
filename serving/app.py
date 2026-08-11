"""The prediction API.

Wraps the offline pipeline behind HTTP without duplicating any of it. Cleaning,
feature building and prediction all run through the same code the training run
used, which is what keeps a served prediction identical to an offline one.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from serving.dependencies import (
    ModelNotLoadedError,
    UnknownCityError,
    build_warnings,
    get_context,
    load_context,
    model_info_payload,
    predict,
)
from serving.schemas import (
    ActualRequest,
    ActualResponse,
    BatchActualRequest,
    BatchActualResponse,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    LoadRequest,
    ModelInfo,
    PredictionResponse,
)
from serving.store import (
    PredictionRecord,
    StoreUnavailableError,
    get_store,
    reset_store,
)
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model once at startup and hold it for the process lifetime.

    A failure here is logged but does not stop the process, so the health
    endpoint stays reachable and reports the problem rather than the container
    dying with no explanation.

    Args:
        app: The application being started.

    Yields:
        None, while the application runs.
    """
    setup_logging()
    logger.info("Starting freight rate API %s", API_VERSION)

    try:
        load_context()
    except ModelNotLoadedError as exc:
        logger.error("Startup could not load a model: %s", exc)

    # Logging is optional. A store that will not connect degrades monitoring,
    # never serving, so a failure here is recorded and the API carries on.
    reset_store().connect()

    yield

    get_store().close()
    logger.info("Shutting down")


app = FastAPI(
    title="Freight Rate Prediction API",
    description=(
        "Prices a freight load from its lane, distance, trailer type, weight "
        "and date. Coordinates are looked up from the city name, so callers "
        "supply only the six fields the model actually needs."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    responses={503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_latency(request: Request, call_next):
    """Time every request and return it in a response header.

    Args:
        request: The incoming request.
        call_next: The next handler in the chain.

    Returns:
        The response, with an X-Process-Time-Ms header added.
    """
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    return response


@app.exception_handler(ModelNotLoadedError)
async def handle_model_not_loaded(request: Request, exc: ModelNotLoadedError):
    """Return 503 rather than 500 when the model is missing.

    The service is reachable but cannot serve, which is what 503 means.

    Args:
        request: The request that failed.
        exc: The raised error.

    Returns:
        A 503 response.
    """
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_type": "model_not_loaded"},
    )


@app.exception_handler(UnknownCityError)
async def handle_unknown_city(request: Request, exc: UnknownCityError):
    """Return 422 with the cities that could not be placed.

    Args:
        request: The request that failed.
        exc: The raised error.

    Returns:
        A 422 response naming the unplaceable cities.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "error_type": "unknown_city"},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Report whether the service can serve predictions.

    Always returns 200 so a probe can tell a reachable but degraded service
    apart from one that is down.

    Returns:
        The current status.
    """
    store_up = get_store().is_available

    try:
        context = get_context()
    except ModelNotLoadedError:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            uptime_seconds=0.0,
            store_available=store_up,
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=context.version,
        uptime_seconds=round(context.uptime_seconds, 1),
        store_available=store_up,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
async def model_info() -> ModelInfo:
    """Describe the loaded model and how it scored during training.

    Returns:
        The model description.
    """
    return ModelInfo(**model_info_payload())


def _records(
    loads: list[LoadRequest],
    rates: list[float],
    warnings_list: list,
    version: str,
    latency_ms: float,
) -> list[PredictionRecord]:
    """Build the rows to log for a set of predictions.

    Args:
        loads: The submitted requests.
        rates: The predicted rates.
        warnings_list: One set of warnings per load.
        version: The serving model version.
        latency_ms: Time taken per load.

    Returns:
        Records ready for the store.
    """
    return [
        PredictionRecord(
            load_id=load.load_id,
            pickup=load.pickup,
            delivery=load.delivery,
            distance=load.distance,
            equipment=load.equipment,
            weight=load.weight,
            load_date=load.date,
            predicted_rate=rate,
            rate_per_mile=rate / load.distance,
            model_version=version,
            unknown_city=warning.unknown_city,
            date_beyond_training=warning.date_beyond_training,
            distance_out_of_range=warning.distance_out_of_range,
            weight_imputed=warning.weight_imputed,
            latency_ms=latency_ms,
        )
        for load, rate, warning in zip(loads, rates, warnings_list)
    ]


@app.post("/actuals", response_model=ActualResponse, tags=["monitoring"])
async def record_actual(request: ActualRequest) -> ActualResponse:
    """Record the rate a load actually went for.

    Outcomes arrive days or weeks after the quote, which is why this is a
    separate endpoint rather than part of the prediction response. Without it
    the model can be monitored for drift but never scored.

    Args:
        request: The load and its confirmed rate.

    Returns:
        Confirmation that the outcome was stored.

    Raises:
        HTTPException: If the store cannot be reached.
    """
    try:
        get_store().record_actual(request.load_id, request.actual_rate, request.source)
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"could not record: {exc}") from exc

    return ActualResponse(
        load_id=request.load_id,
        actual_rate=request.actual_rate,
        recorded=True,
    )


@app.post("/actuals/batch", response_model=BatchActualResponse, tags=["monitoring"])
async def record_actuals(request: BatchActualRequest) -> BatchActualResponse:
    """Record many outcomes in one call.

    Args:
        request: The outcomes to store.

    Returns:
        How many were written.

    Raises:
        HTTPException: If the store cannot be reached.
    """
    rows = [(a.load_id, a.actual_rate, a.source) for a in request.actuals]

    try:
        recorded = get_store().record_actuals(rows)
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"could not record: {exc}") from exc

    return BatchActualResponse(recorded=recorded, failed=len(rows) - recorded)


@app.get("/metrics/performance", tags=["monitoring"])
async def performance(days: int = Query(default=30, ge=1, le=365)) -> dict:
    """Report how the model has done on traffic that has outcomes.

    Args:
        days: How far back to look.

    Returns:
        The performance snapshot, including how much of the traffic could be
        scored at all.

    Raises:
        HTTPException: If the store cannot be reached.
    """
    from monitoring.performance import snapshot

    try:
        return snapshot(get_store(), days).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"metrics unavailable: {exc}") from exc


@app.get("/metrics/traffic", tags=["monitoring"])
async def traffic(days: int = Query(default=7, ge=1, le=365)) -> dict:
    """Describe recent traffic, whether or not outcomes have arrived.

    Available immediately rather than waiting on feedback, which makes it the
    first place to look when something changes.

    Args:
        days: How far back to look.

    Returns:
        Volume, warning rates, latency and the predicted rate distribution.

    Raises:
        HTTPException: If the store cannot be reached.
    """
    from monitoring.performance import traffic_summary

    try:
        return traffic_summary(get_store(), days)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"metrics unavailable: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
async def predict_one(load: LoadRequest, background: BackgroundTasks) -> PredictionResponse:
    """Price a single load.

    Args:
        load: The load to price.

    Returns:
        The predicted rate, with any warnings that apply.

    Raises:
        HTTPException: If the prediction itself fails.
    """
    context = get_context()
    started = time.perf_counter()

    try:
        rate = predict([load], context)[0]
    except UnknownCityError:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

    warnings = build_warnings([load], context)[0]
    elapsed = (time.perf_counter() - started) * 1000

    if warnings.any:
        logger.info("Prediction served with warnings: %s", warnings.model_dump())

    # Queued after the response is sent, so logging never adds to the latency
    # the caller sees.
    background.add_task(
        get_store().log_predictions,
        _records([load], [rate], [warnings], context.version, elapsed),
    )

    return PredictionResponse(
        predicted_rate=round(rate, 2),
        rate_per_mile=round(rate / load.distance, 3),
        model_version=context.version,
        warnings=warnings,
        latency_ms=round(elapsed, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
async def predict_batch(request: BatchRequest, background: BackgroundTasks) -> BatchResponse:
    """Price several loads in one call.

    Far cheaper per load than repeated single calls, because the feature build
    and the model run once across the whole batch.

    Args:
        request: The loads to price.

    Returns:
        One prediction per load, in the order submitted.

    Raises:
        HTTPException: If the prediction itself fails.
    """
    context = get_context()
    started = time.perf_counter()

    try:
        rates = predict(request.loads, context)
    except UnknownCityError:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

    warnings = build_warnings(request.loads, context)
    elapsed = (time.perf_counter() - started) * 1000
    per_load = elapsed / len(rates)

    predictions = [
        PredictionResponse(
            predicted_rate=round(rate, 2),
            rate_per_mile=round(rate / load.distance, 3),
            model_version=context.version,
            warnings=warning,
            latency_ms=round(per_load, 2),
        )
        for rate, load, warning in zip(rates, request.loads, warnings)
    ]

    flagged = sum(1 for w in warnings if w.any)
    logger.info("Priced %s loads, %s with warnings", len(rates), flagged)

    background.add_task(
        get_store().log_predictions,
        _records(request.loads, rates, warnings, context.version, per_load),
    )

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        latency_ms=round(elapsed, 2),
    )


@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Point a browser at the docs.

    Returns:
        A short pointer to the interactive documentation.
    """
    return {
        "service": "Freight Rate Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }