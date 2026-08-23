"""Generating the outcomes the replay needs.

The validation file has no rates in it, so a replay has nothing to score
against unless outcomes are manufactured. Everything here is synthetic and is
labelled as such wherever it surfaces.

The point is not to make the model look good. It is to give monitoring
something real to detect. The default scenario applies a December uplift the
model was never trained on, because training stops on 31 October and the ten
months before it contain no peak season. Error should therefore grow through
December, and the dashboard should show it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import numpy as np

# Noise on every outcome, whatever the scenario. Real quoted and settled rates
# differ by a few percent even when nothing is wrong.
BASE_NOISE = 0.04

# Not every load comes back. A broker confirms most but not all of them.
DEFAULT_FEEDBACK_RATE = 0.65


@dataclass(frozen=True)
class Scenario:
    """How synthetic outcomes differ from what the model predicts.

    Args:
        name: Identifier used in logs and on the command line.
        description: What the scenario is meant to demonstrate.
        noise: Relative standard deviation applied to every outcome.
        december_uplift: Peak season lift applied through December, reaching
            this fraction by the 31st. The model has never seen a December, so
            this is the drift it cannot anticipate.
        unknown_city_penalty: Extra error on loads involving a city the model
            has no history for.
        shock_start: Date a sudden market move begins, or None.
        shock_size: Size of that move as a fraction.
    """

    name: str
    description: str
    noise: float = BASE_NOISE
    december_uplift: float = 0.0
    unknown_city_penalty: float = 0.0
    shock_start: Date | None = None
    shock_size: float = 0.0


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="No drift. Outcomes differ from predictions by noise alone.",
        noise=BASE_NOISE,
    ),
    "peak_season": Scenario(
        name="peak_season",
        description=(
            "Rates climb through December for peak season. The model trained "
            "on January to October and has never seen this, so error grows as "
            "the month goes on."
        ),
        noise=BASE_NOISE,
        december_uplift=0.12,
        unknown_city_penalty=0.05,
    ),
    "shock": Scenario(
        name="shock",
        description=(
            "A sudden market move partway through the replay, of the kind a "
            "capacity crunch or a fuel spike would cause."
        ),
        noise=BASE_NOISE,
        shock_start=Date(2025, 12, 1),
        shock_size=0.20,
    ),
}


class OutcomeGenerator:
    """Turns predictions into the rates those loads supposedly settled at.

    Args:
        scenario: Which behaviour to simulate.
        feedback_rate: Share of loads that ever report an outcome.
        seed: Random seed, so a replay can be repeated exactly.
    """

    def __init__(
        self,
        scenario: Scenario,
        feedback_rate: float = DEFAULT_FEEDBACK_RATE,
        seed: int = 42,
    ) -> None:
        self.scenario = scenario
        self.feedback_rate = feedback_rate
        self.rng = np.random.default_rng(seed)

    def _multiplier(self, load_date: Date, unknown_city: bool) -> float:
        """Work out how far the true rate sits from the predicted one.

        Args:
            load_date: When the load moves.
            unknown_city: Whether the model had history for both cities.

        Returns:
            A multiplier to apply to the predicted rate.
        """
        multiplier = 1.0

        # Ramps through December rather than switching on, so the dashboard
        # shows error growing instead of jumping.
        if self.scenario.december_uplift and load_date.month == 12:
            progress = load_date.day / 31.0
            multiplier *= 1.0 + self.scenario.december_uplift * progress

        if self.scenario.shock_start and load_date >= self.scenario.shock_start:
            multiplier *= 1.0 + self.scenario.shock_size

        # An unfamiliar city is priced from position alone, with no local
        # history, so its outcomes sit further from the prediction.
        if unknown_city and self.scenario.unknown_city_penalty:
            multiplier *= 1.0 + self.rng.normal(0, self.scenario.unknown_city_penalty)

        return multiplier * self.rng.normal(1.0, self.scenario.noise)

    def outcome_for(
        self,
        predicted_rate: float,
        load_date: Date,
        unknown_city: bool = False,
    ) -> float | None:
        """Produce the settled rate for one load.

        Args:
            predicted_rate: What the model quoted.
            load_date: When the load moves.
            unknown_city: Whether either city was unfamiliar to the model.

        Returns:
            The synthetic settled rate, or None when this load never reports
            an outcome.
        """
        if self.rng.random() > self.feedback_rate:
            return None

        rate = predicted_rate * self._multiplier(load_date, unknown_city)
        return round(max(rate, 1.0), 2)
