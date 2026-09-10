# Architecture and design decisions

## Data flow

1. `data.py` creates a deterministic, privacy-safe dataset and train/test split.
2. `models.py` standardises features, fits PCA and logistic regression, segments customers, and computes metrics.
3. `experiment.py` evaluates a documented conversion experiment under beta-binomial uncertainty.
4. `database.py` persists results transactionally in SQLite.
5. `pipeline.py` orchestrates the study and writes reproducible artifacts.
6. `api.py` exposes only read operations through a dependency-free HTTP server.
7. `cli.py` gives stable entry points for experiments and serving.

## Key decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| Synthetic data | Public, deterministic, privacy-safe | Cannot prove external validity |
| Models implemented with NumPy | Mathematics remains inspectable | Fewer production optimisations |
| SQLite | Portable transactions and SQL evidence | Not designed for horizontal scale |
| Standard-library HTTP server | Minimal runtime and transparent routing | Intentionally small API surface |
| Fixed random seeds | Exact reproduction and testability | A single run hides seed sensitivity |

## Complexity

- Logistic regression: $O(ind)$ for $i$ iterations, $n$ records, and $d$ features.
- PCA covariance and eigendecomposition: $O(nd^2+d^3)$.
- K-means: $O(inkd)$ for $i$ iterations and $k$ clusters.
- Database lookup by primary key: expected $O(\log n)$ B-tree access.

## Reliability and security scope

Database mutations occur inside context-managed transactions. Parameters are bound rather than interpolated into SQL. The API is read-only and returns JSON with explicit status codes. A production version would add authentication, TLS termination, request limits, structured logging, migrations, and monitoring.
