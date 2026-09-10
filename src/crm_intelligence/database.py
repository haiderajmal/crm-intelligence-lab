"""Small transactional persistence layer using the Python standard library."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from .data import CustomerDataset


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    recency_days REAL NOT NULL,
    purchase_frequency REAL NOT NULL,
    average_order_value REAL NOT NULL,
    engagement_score REAL NOT NULL,
    support_contacts REAL NOT NULL,
    discount_sensitivity REAL NOT NULL,
    churned INTEGER NOT NULL CHECK (churned IN (0, 1)),
    churn_probability REAL NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    segment INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    result_json TEXT NOT NULL
);
"""


class CRMDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialise(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def replace_customers(
        self, dataset: CustomerDataset, probabilities: np.ndarray, segments: np.ndarray
    ) -> None:
        rows = [
            (
                int(customer_id),
                *[float(value) for value in features],
                int(churned),
                float(probability),
                int(segment),
            )
            for customer_id, features, churned, probability, segment in zip(
                dataset.customer_ids,
                dataset.features,
                dataset.churned,
                probabilities,
                segments,
                strict=True,
            )
        ]
        with self.connect() as connection:
            connection.execute("DELETE FROM customers")
            connection.executemany(
                """
                INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def add_experiment(self, result: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO experiments(result_json) VALUES (?)",
                (json.dumps(result, allow_nan=False),),
            )

    def get_customer(self, customer_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        return dict(row) if row else None

    def latest_experiment(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT experiment_id, created_at, result_json FROM experiments "
                "ORDER BY experiment_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {
            "experiment_id": row["experiment_id"],
            "created_at": row["created_at"],
            **json.loads(row["result_json"]),
        }

    def customer_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0])
