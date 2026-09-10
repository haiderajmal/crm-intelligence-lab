"""Bayesian evaluation of a binary CRM intervention."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ABResult:
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float | None
    probability_treatment_better: float
    credible_interval_low: float
    credible_interval_high: float

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def bayesian_ab_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    *,
    draws: int = 60_000,
    seed: int = 42,
) -> ABResult:
    """Compare conversion rates with independent Beta(1, 1) priors."""
    values = (control_successes, control_total, treatment_successes, treatment_total)
    if any(v < 0 for v in values):
        raise ValueError("counts cannot be negative")
    if control_successes > control_total or treatment_successes > treatment_total:
        raise ValueError("successes cannot exceed totals")
    if min(control_total, treatment_total, draws) == 0:
        raise ValueError("totals and draws must be positive")

    rng = np.random.default_rng(seed)
    control = rng.beta(
        1 + control_successes, 1 + control_total - control_successes, draws
    )
    treatment = rng.beta(
        1 + treatment_successes, 1 + treatment_total - treatment_successes, draws
    )
    difference = treatment - control
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    low, high = np.quantile(difference, [0.025, 0.975])
    return ABResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_lift=treatment_rate - control_rate,
        relative_lift=(treatment_rate / control_rate - 1.0) if control_rate else None,
        probability_treatment_better=float(np.mean(difference > 0)),
        credible_interval_low=float(low),
        credible_interval_high=float(high),
    )
