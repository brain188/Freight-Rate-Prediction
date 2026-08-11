"""Replaying the validation set at the API as if it were live traffic.

Without this the monitoring tables are empty and there is nothing to look at.
The replay walks November and December one simulated day at a time, prices
that day's loads, and reports outcomes back on a delay so the feedback lag is
real rather than assumed.

Two things make this worth watching rather than just filling a table. The
prediction window contains eight cities the model has never seen, so the
unknown city rate climbs on genuine data. And the default scenario applies a
December uplift the model was never trained on, so accuracy degrades as the
month goes on and the monitoring should catch it.

    python simulator/replay.py --speed 30
    python simulator/replay.py --scenario shock --feedback-delay 5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.outcomes import SCENARIOS, OutcomeGenerator
from src.config import load_config
from src.logger import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_API = "http://localhost:8000"
BATCH_SIZE = 250
RULE = "-" * 78


class ReplayError(RuntimeError):
    """Raised when the replay cannot run."""


def parse_args() -> argparse.Namespace:
    """Read the command line options.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Replay the validation set at the API as simulated live traffic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python simulator/replay.py\n"
            "  python simulator/replay.py --speed 30 --scenario peak_season\n"
            "  python simulator/replay.py --days 14 --no-outcomes\n"
        ),
    )
    parser.add_argument("--api-url", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    parser.add_argument("--speed", type=float, default=20.0,
                        help="simulated days per real second (default: 20)")
    parser.add_argument("--days", type=int, default=None,
                        help="stop after this many simulated days")
    parser.add_argument("--scenario", default="peak_season", choices=sorted(SCENARIOS),
                        help="how synthetic outcomes behave (default: peak_season)")
    parser.add_argument("--feedback-delay", type=int, default=3,
                        help="simulated days before an outcome is reported (default: 3)")
    parser.add_argument("--feedback-rate", type=float, default=0.65,
                        help="share of loads that ever report an outcome (default: 0.65)")
    parser.add_argument("--no-outcomes", action="store_true",
                        help="send predictions only, so nothing can be scored")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_traffic(days: int | None) -> pd.DataFrame:
    """Read the validation set and group it by date.

    Args:
        days: Stop after this many distinct dates, or None for all of them.

    Returns:
        The loads to replay, sorted by date.

    Raises:
        ReplayError: If the validation file is missing.
    """
    config = load_config(create_dirs=False)

    if not config.paths.validation.is_file():
        raise ReplayError(f"validation file not found: {config.paths.validation}")

    frame = pd.read_csv(config.paths.validation, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    if days:
        keep = sorted(frame["date"].dt.date.unique())[:days]
        frame = frame[frame["date"].dt.date.isin(keep)]

    return frame


def to_payload(row: pd.Series) -> dict:
    """Turn one row of the validation set into a request body.

    Coordinates are passed through because the file carries them, which is what
    lets the eight unfamiliar cities be priced at all.

    Args:
        row: One validation load.

    Returns:
        The request body.
    """
    payload = {
        "load_id": str(row["load_id"]),
        "pickup": str(row["pickup"]),
        "delivery": str(row["delivery"]),
        "distance": float(row["distance"]),
        "equipment": str(row["equipment"]),
        "date": str(pd.Timestamp(row["date"]).date()),
    }

    if pd.notna(row.get("weight")):
        payload["weight"] = float(row["weight"])

    for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        if pd.notna(row.get(column)):
            payload[column] = float(row[column])

    return payload


def wait_for_api(client: httpx.Client, attempts: int = 10) -> dict:
    """Confirm the API is up and has a model before sending anything.

    Args:
        client: The HTTP client.
        attempts: How many times to retry.

    Returns:
        The health response.

    Raises:
        ReplayError: If the API never becomes ready.
    """
    for attempt in range(attempts):
        try:
            health = client.get("/health", timeout=5.0).json()
            if health.get("model_loaded"):
                return health
            logger.warning("API is up but has no model loaded")
        except httpx.HTTPError:
            pass

        time.sleep(1.0 + attempt * 0.5)

    raise ReplayError(
        "could not reach a ready API. Start it with:\n"
        "  uvicorn serving.app:app --port 8000"
    )


def send_day(client: httpx.Client, loads: pd.DataFrame) -> list[dict]:
    """Price one simulated day of loads.

    Args:
        client: The HTTP client.
        loads: That day's loads.

    Returns:
        One record per load, pairing the request with its prediction.
    """
    results = []

    for start in range(0, len(loads), BATCH_SIZE):
        chunk = loads.iloc[start:start + BATCH_SIZE]
        payloads = [to_payload(row) for _, row in chunk.iterrows()]

        try:
            response = client.post("/predict/batch", json={"loads": payloads}, timeout=60.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Batch failed: %s", exc)
            continue

        for payload, prediction in zip(payloads, response.json()["predictions"]):
            results.append({
                "load_id": payload["load_id"],
                "load_date": pd.Timestamp(payload["date"]).date(),
                "predicted_rate": prediction["predicted_rate"],
                "unknown_city": prediction["warnings"]["unknown_city"],
            })

    return results


def send_outcomes(client: httpx.Client, outcomes: list[tuple[str, float]]) -> int:
    """Report settled rates back to the API.

    Args:
        client: The HTTP client.
        outcomes: Load identifiers paired with their settled rate.

    Returns:
        How many were accepted.
    """
    if not outcomes:
        return 0

    payload = {
        "actuals": [
            {"load_id": load_id, "actual_rate": rate, "source": "replay"}
            for load_id, rate in outcomes
        ]
    }

    try:
        response = client.post("/actuals/batch", json=payload, timeout=60.0)
        response.raise_for_status()
        return int(response.json()["recorded"])
    except httpx.HTTPError as exc:
        logger.error("Could not report %s outcomes: %s", len(outcomes), exc)
        return 0


def run(args: argparse.Namespace) -> int:
    """Run the replay.

    Args:
        args: Parsed command line arguments.

    Returns:
        0 on success, 1 on failure.
    """
    scenario = SCENARIOS[args.scenario]
    generator = OutcomeGenerator(scenario, args.feedback_rate, args.seed)

    traffic = load_traffic(args.days)
    days = sorted(traffic["date"].dt.date.unique())

    print(f"\nReplay\n{RULE}")
    print(f"loads          {len(traffic):,} across {len(days)} days")
    print(f"period         {days[0]} to {days[-1]}")
    print(f"scenario       {scenario.name}")
    print(f"               {scenario.description}")
    print(f"outcomes       {'disabled' if args.no_outcomes else f'{args.feedback_rate:.0%} reported after {args.feedback_delay} days'}")
    print(f"speed          {args.speed:g} simulated days per second")
    print(RULE)

    # Outcomes are held until their reporting day, so feedback arrives late in
    # simulated time exactly as it would in reality.
    pending: dict[Date, list[tuple[str, float]]] = defaultdict(list)

    totals = {"priced": 0, "outcomes": 0, "unknown_city": 0}
    delay = 1.0 / args.speed if args.speed > 0 else 0.0

    with httpx.Client(base_url=args.api_url) as client:
        health = wait_for_api(client)
        print(f"API ready, model {health['model_version']}, "
              f"logging {'on' if health.get('store_available') else 'OFF'}\n")

        print(f"{'day':<12}{'priced':>8}{'unknown':>9}{'outcomes':>10}  {'running MAPE':>12}")

        for day in days:
            batch = traffic[traffic["date"].dt.date == day]
            results = send_day(client, batch)

            totals["priced"] += len(results)
            flagged = sum(1 for r in results if r["unknown_city"])
            totals["unknown_city"] += flagged

            if not args.no_outcomes:
                report_on = day + timedelta(days=args.feedback_delay)
                for record in results:
                    rate = generator.outcome_for(
                        record["predicted_rate"], record["load_date"], record["unknown_city"]
                    )
                    if rate is not None:
                        pending[report_on].append((record["load_id"], rate))

            due = pending.pop(day, [])
            sent = send_outcomes(client, due) if due else 0
            totals["outcomes"] += sent

            mape = _current_mape(client)
            mape_text = f"{mape:.2f}%" if mape is not None else "waiting"
            print(f"{day!s:<12}{len(results):>8}{flagged:>9}{sent:>10}  {mape_text:>12}")

            if delay:
                time.sleep(delay)

        # Anything still held back would never be reported otherwise.
        remaining = [item for items in pending.values() for item in items]
        if remaining:
            totals["outcomes"] += send_outcomes(client, remaining)

        _print_summary(client, totals)

    return 0


def _current_mape(client: httpx.Client) -> float | None:
    """Read the running error from the API.

    Args:
        client: The HTTP client.

    Returns:
        MAPE over the last 90 days, or None when nothing can be scored yet.
    """
    try:
        body = client.get("/metrics/performance", params={"days": 90}, timeout=10.0).json()
    except (httpx.HTTPError, ValueError):
        return None

    return body.get("metrics", {}).get("mape") if body.get("n_scored") else None


def _print_summary(client: httpx.Client, totals: dict) -> None:
    """Print what the replay produced and what monitoring made of it.

    Args:
        client: The HTTP client.
        totals: Running counts from the replay.
    """
    print(f"\n{RULE}")
    print(f"priced         {totals['priced']:,}")
    print(f"unknown city   {totals['unknown_city']:,} "
          f"({totals['unknown_city'] / max(totals['priced'], 1):.1%})")
    print(f"outcomes       {totals['outcomes']:,}")

    try:
        performance = client.get("/metrics/performance", params={"days": 90}).json()
        traffic = client.get("/metrics/traffic", params={"days": 90}).json()
    except (httpx.HTTPError, ValueError):
        print("\nCould not read metrics back from the API.")
        return

    print(f"\nMonitoring\n{RULE}")
    print(f"coverage             {performance['coverage']:.1%} "
          f"({performance['n_scored']:,} of {performance['n_predictions']:,} scored)")
    print(f"reliable             {performance['is_reliable']}")

    for name in ("rmse", "mae", "mape", "r2", "bias"):
        if name in performance.get("metrics", {}):
            value = performance["metrics"][name]
            unit = "%" if name == "mape" else ""
            print(f"{name:<21}{value:,.4f}{unit}")

    print(f"unknown city rate    {traffic.get('unknown_city_rate', 0):.1%}")
    print(f"p95 latency          {traffic.get('p95_latency_ms', 0):.2f} ms")
    print("\nNote: outcomes are synthetic. See simulator/outcomes.py.")


def main() -> int:
    """Entry point.

    Returns:
        0 on success, 1 on failure.
    """
    args = parse_args()
    setup_logging(level="WARNING")

    # httpx logs every request at INFO, which buries the replay's own output.
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        return run(args)
    except ReplayError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())