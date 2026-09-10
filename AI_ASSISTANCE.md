# AI assistance disclosure

## Scope

The initial version of this repository was produced on 10 September 2026 with substantial assistance from OpenAI Codex, based on direction from Muhammad Haider Ali. Assistance covered:

- project selection and software architecture;
- initial Python implementation;
- automated tests and continuous-integration configuration;
- scientific and technical documentation; and
- local execution and visual inspection of generated plots.

## Human responsibility

Muhammad Haider Ali is the repository owner and is responsible for evaluating the code, confirming that it is appropriate for its intended use, and complying with university rules. The repository must not be described as entirely human-generated or as evidence of skills that the owner cannot independently demonstrate.

Before using this project in an application, the owner should:

1. run the complete workflow locally;
2. explain PCA, logistic loss and its gradient, k-means, ROC-AUC, and the beta-binomial model without relying on generated text;
3. trace the data flow from generation through SQLite and the API;
4. make and document at least one substantive personal extension; and
5. retain a transparent commit history for that extension.

Good extension candidates include repeated-seed stability analysis, probability calibration, temporal validation, or a small interface that consumes the JSON API.

## Validation record for the initial version

- Python test suite: 10 tests passed.
- End-to-end run: 1,200 synthetic customers, seed 42.
- Held-out ROC-AUC: 0.687.
- Training-selected decision threshold: 0.285.
- Bayesian probability that treatment improves conversion: 0.963.
- Outputs visually inspected: optimisation loss curve and held-out ROC curve.

Synthetic results demonstrate the workflow only and are not claims about a real business or customer population.
