# Transparent Customer Intelligence for CRM Decision Support

**Author:** Muhammad Haider Ali  
**Repository:** `haiderajmal/crm-intelligence-lab`

## Abstract

This study develops a transparent and reproducible customer-intelligence pipeline for a retail CRM setting. It asks whether interpretable mathematical methods can identify churn risk, derive behavioural segments, and quantify the uncertainty of a proposed engagement intervention. Because real customer data involve privacy and ownership constraints, the study uses a deterministic synthetic dataset. Principal component analysis (PCA), logistic regression trained by gradient descent, k-means clustering, and a Bayesian beta-binomial A/B test are implemented directly with NumPy. A held-out test design, fixed random seeds, persistent outputs, and automated tests make the results reproducible. The project demonstrates a technical bridge between customer-engagement research and operational analytics without claiming that synthetic results generalise to a real retailer.

## 1. Research question and hypotheses

**RQ:** Can an auditable analytics workflow support CRM decisions by estimating churn risk, identifying engagement segments, and evaluating an intervention under uncertainty?

- H1: Behavioural variables contain sufficient signal for a logistic model to discriminate churned from retained customers above a random ROC-AUC baseline of 0.5.
- H2: A treatment conversion rate of 13.2% versus a control rate of 10.6% yields high posterior probability that treatment is better under weak Beta(1,1) priors.

## 2. Data and ethics

The generator creates six continuous or count-like features: recency, purchase frequency, average order value, engagement, support contacts, and discount sensitivity. Churn is sampled from a documented logistic data-generating process. Customer identifiers are sequential and fictitious.

Synthetic data improve reproducibility and avoid personal-data disclosure, but they cannot establish effectiveness in a real population. Any deployment would require a lawful basis for processing, purpose limitation, access control, bias assessment, monitoring, and human oversight.

## 3. Methods

### 3.1 Standardisation and PCA

For feature vector $x$, standardisation uses $z_j=(x_j-\mu_j)/\sigma_j$. PCA eigendecomposes the sample covariance matrix

$$C=\frac{1}{n-1}Z^T Z$$

and projects standardised observations onto the eigenvectors associated with the two largest eigenvalues. The projected coordinates support visualisable clustering and explicitly demonstrate linear algebra.

### 3.2 Logistic regression and calculus

The churn probability is

$$P(y=1\mid x)=\sigma(w^T x+b),\qquad \sigma(t)=\frac{1}{1+e^{-t}}.$$

The optimiser minimises binary cross-entropy plus L2 regularisation. Its weight gradient is

$$\nabla_w L=\frac{1}{n}X^T(\hat y-y)+\lambda w.$$

Weights are updated by $w_{k+1}=w_k-\eta\nabla_wL$. Convergence stops when the absolute loss improvement falls below a fixed tolerance.

### 3.3 Segmentation algorithm

K-means alternates assignment to the nearest centroid and centroid recomputation. Each iteration costs $O(nkd)$ for $n$ customers, $k$ segments, and $d$ projected dimensions. A maximum-iteration bound and movement tolerance guarantee termination.

### 3.4 Bayesian A/B evaluation

With a Beta(1,1) prior and binomial observations, each posterior remains beta distributed. Monte Carlo draws estimate $P(p_T>p_C)$ and the 95% credible interval of $p_T-p_C$. A fixed seed makes the approximation reproducible.

### 3.5 Validation

The churn model is evaluated only on a held-out 25% test set using accuracy, precision, recall, F1, and ROC-AUC. Because churn is imbalanced, the decision threshold is selected by maximising F1 on the training partition only; the held-out labels never influence that choice. Automated tests cover invariants, known ranking cases, database round trips, and deterministic integration outputs.

## 4. Reproduction protocol

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
crm-lab run --customers 2000 --seed 42 --output artifacts
```

The command records machine-readable metrics, a SQLite database, convergence and ROC plots, and a run-specific Markdown summary.

## 5. Interpretation

The primary criterion for H1 is held-out ROC-AUC above 0.5. The primary criterion for H2 is the posterior probability that treatment outperforms control, interpreted together with the credible interval for absolute lift. Predictive discrimination does not imply causal explanation: model coefficients recover associations under the synthetic process. Likewise, the A/B calculation supports a causal reading only when treatment assignment is random and the experimental protocol is valid.

## 6. Limitations and future work

- Synthetic observations omit missingness, temporal drift, channel interactions, and selection bias.
- Random initialisation can affect k-means segments; repeated initialisations and stability analysis would strengthen it.
- A threshold of 0.5 is illustrative; operational thresholds should reflect intervention costs and calibration.
- Real-world validation should use time-aware splitting, pre-registered metrics, fairness slices, and privacy-preserving governance.
- Future work can connect the conceptual Payne and Frow CRM processes to measurable events and decision logs.

## 7. Conclusion

The project provides an inspectable link between mathematical foundations, software engineering, and CRM research. Its main contribution is not a production claim but a reproducible demonstration of how customer-engagement hypotheses can be expressed, tested, stored, and exposed through a small software system.
