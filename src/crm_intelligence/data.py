"""Deterministic synthetic CRM data for reproducible public experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "recency_days",
    "purchase_frequency",
    "average_order_value",
    "engagement_score",
    "support_contacts",
    "discount_sensitivity",
)


@dataclass(frozen=True)
class CustomerDataset:
    customer_ids: np.ndarray
    features: np.ndarray
    churned: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def generate_customers(n_customers: int = 1_200, seed: int = 42) -> CustomerDataset:
    """Generate plausible, non-identifying retail customer behaviour."""
    if n_customers < 100:
        raise ValueError("n_customers must be at least 100")

    rng = np.random.default_rng(seed)
    recency = np.clip(rng.gamma(2.2, 18.0, n_customers), 1, 180)
    frequency = np.clip(rng.poisson(5.5, n_customers) + 1, 1, 30)
    order_value = np.clip(rng.lognormal(3.65, 0.45, n_customers), 8, 250)
    engagement = rng.beta(2.4, 2.0, n_customers)
    support = np.clip(rng.poisson(1.2, n_customers), 0, 8)
    discount = rng.beta(2.0, 2.5, n_customers)

    linear_signal = (
        -0.75
        + 0.022 * recency
        - 0.16 * frequency
        - 0.006 * order_value
        - 2.0 * engagement
        + 0.28 * support
        + 0.9 * discount
    )
    churn_probability = sigmoid(linear_signal)
    churned = rng.binomial(1, churn_probability).astype(int)
    features = np.column_stack(
        [recency, frequency, order_value, engagement, support, discount]
    ).astype(float)
    return CustomerDataset(np.arange(1, n_customers + 1), features, churned)


def train_test_split(
    dataset: CustomerDataset, test_fraction: float = 0.25, seed: int = 42
) -> tuple[CustomerDataset, CustomerDataset]:
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset.customer_ids))
    cut = int(len(indices) * (1.0 - test_fraction))

    def subset(idx: np.ndarray) -> CustomerDataset:
        return CustomerDataset(
            dataset.customer_ids[idx], dataset.features[idx], dataset.churned[idx]
        )

    return subset(indices[:cut]), subset(indices[cut:])
