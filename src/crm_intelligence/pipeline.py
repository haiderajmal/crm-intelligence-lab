"""End-to-end experiment orchestration and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import generate_customers, train_test_split
from .database import CRMDatabase
from .experiment import bayesian_ab_test
from .models import (
    KMeans,
    LogisticRegressionGD,
    PrincipalComponentAnalysis,
    StandardScaler,
    classification_metrics,
    roc_curve,
    select_f1_threshold,
)


def run_pipeline(output: str | Path = "artifacts", customers: int = 1_200, seed: int = 42) -> dict:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = generate_customers(customers, seed)
    train, test = train_test_split(dataset, seed=seed)

    scaler = StandardScaler().fit(train.features)
    train_scaled = scaler.transform(train.features)
    test_scaled = scaler.transform(test.features)
    all_scaled = scaler.transform(dataset.features)

    model = LogisticRegressionGD().fit(train_scaled, train.churned)
    decision_threshold = select_f1_threshold(train.churned, model.predict_proba(train_scaled))
    test_probabilities = model.predict_proba(test_scaled)
    all_probabilities = model.predict_proba(all_scaled)
    metrics = classification_metrics(test.churned, test_probabilities, decision_threshold)

    pca = PrincipalComponentAnalysis(2).fit(train_scaled)
    projected = pca.transform(all_scaled)
    segments = KMeans(n_clusters=3, seed=seed).fit(projected).predict(projected)

    ab_result = bayesian_ab_test(106, 1_000, 132, 1_000, seed=seed)
    result = {
        "seed": seed,
        "customers": customers,
        "train_customers": len(train.customer_ids),
        "test_customers": len(test.customer_ids),
        "model": metrics,
        "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
        "optimisation_iterations": len(model.loss_history_),
        "ab_test": ab_result.as_dict(),
        "segment_sizes": {
            str(i): int(np.sum(segments == i)) for i in range(3)
        },
    }

    database = CRMDatabase(output_path / "crm_lab.sqlite3")
    database.initialise()
    database.replace_customers(dataset, all_probabilities, segments)
    database.add_experiment(result["ab_test"])
    (output_path / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_loss(model.loss_history_, output_path / "loss_curve.png")
    _plot_roc(test.churned, test_probabilities, metrics["roc_auc"], output_path / "roc_curve.png")
    _write_summary(result, output_path / "run_summary.md")
    return result


def _plot_loss(losses: list[float], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(losses, color="#2563eb", linewidth=2)
    axis.set(title="Logistic Regression Convergence", xlabel="Iteration", ylabel="Regularised log loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_roc(targets: np.ndarray, scores: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr = roc_curve(targets, scores)
    figure, axis = plt.subplots(figsize=(5.5, 5.2))
    axis.plot(fpr, tpr, color="#0f766e", linewidth=2, label=f"Model (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="#64748b", label="Random baseline")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out ROC curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_summary(result: dict, path: Path) -> None:
    model = result["model"]
    ab = result["ab_test"]
    text = f"""# Reproducible run summary

- Random seed: `{result['seed']}`
- Customers: `{result['customers']}`
- Held-out ROC-AUC: `{model['roc_auc']:.3f}`
- Held-out F1: `{model['f1']:.3f}`
- Optimisation iterations: `{result['optimisation_iterations']}`
- P(treatment better than control): `{ab['probability_treatment_better']:.3f}`
- Estimated absolute conversion lift: `{ab['absolute_lift']:.3f}`
- 95% credible interval for lift: `[{ab['credible_interval_low']:.3f}, {ab['credible_interval_high']:.3f}]`

These results use synthetic data and demonstrate the method; they are not business claims.
"""
    path.write_text(text, encoding="utf-8")
