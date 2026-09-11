# CRM Customer Intelligence Lab

[![CI](https://github.com/haiderajmal/crm-intelligence-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/haiderajmal/crm-intelligence-lab/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

A reproducible Python research project for customer-engagement analytics. It turns a synthetic retail CRM dataset into an auditable workflow: data validation, dimensionality reduction, churn modelling, customer segmentation, Bayesian A/B testing, SQLite persistence, and a small REST API.

The project extends the ideas in Muhammad Haider Ali's MSc dissertation on integrating CRM models and analytics for customer engagement. Synthetic data are used deliberately so the repository is public, reproducible, and free of customer-identification risk.

## What it demonstrates

| Area | Evidence in this repository |
|---|---|
| Probability and statistics | Synthetic sampling, Bayesian beta-binomial A/B test, confidence metrics, ROC-AUC |
| Linear algebra | Standardisation, covariance matrix, eigendecomposition, PCA |
| Calculus | Logistic-loss gradient derivation and gradient-descent optimiser |
| Algorithms and data structures | Vectorised ranking, k-means clustering, convergence criteria, complexity notes |
| Practical computer science | Typed modules, SQLite transactions, HTTP API, CLI, tests, Docker, CI |
| Scientific work | Stated question, methodology, controlled seeds, baselines, limitations, reproducible outputs |

> This portfolio provides evidence of programming and scientific practice. It does not replace the Bachelor-level ECTS or transcript evidence required by a university.

## Research question

Can a transparent, reproducible pipeline identify customers at elevated churn risk and useful engagement segments while quantifying whether a proposed CRM intervention improves conversion?

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
crm-lab run --customers 1200 --seed 42
crm-lab serve --port 8000
```

Then visit:

```text
http://localhost:8000/health
http://localhost:8000/customers/1
http://localhost:8000/experiments/latest
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

No external dataset is required. The pipeline writes only generated artifacts to `artifacts/`.

## Reproduce the study

```bash
crm-lab run --customers 2000 --seed 42 --output artifacts
```

Expected outputs:

- `artifacts/crm_lab.sqlite3`: customers, churn scores, segments, and experiment result
- `artifacts/metrics.json`: machine-readable experiment metrics
- `artifacts/loss_curve.png`: optimisation convergence
- `artifacts/roc_curve.png`: held-out discrimination performance
- `artifacts/run_summary.md`: concise run-specific result summary

## Repository map

```text
src/crm_intelligence/   analytics, database, API, and CLI
tests/                  deterministic unit and integration tests
docs/                   scientific report and architecture notes
.github/workflows/      continuous integration
```

## Model transparency

The logistic regression, PCA, k-means, ROC-AUC, and Bayesian A/B calculation are implemented directly with NumPy. This makes the mathematics inspectable rather than hiding it behind a modelling library. The project is educational and must not be used for automated decisions about real customers without data-governance, fairness, privacy, and calibration reviews.

## Author

Muhammad Haider Ali - [GitHub profile](https://github.com/haiderajmal)

## Development disclosure

This portfolio was developed with substantial assistance from OpenAI Codex under the repository owner's direction. AI assistance included initial architecture, source code, tests, and documentation. The owner is responsible for reviewing, running, understanding, and modifying the work before presenting it as evidence of personal programming ability. See [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for the detailed disclosure and validation record.

