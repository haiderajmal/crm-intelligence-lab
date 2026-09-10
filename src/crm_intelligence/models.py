"""Core mathematical models implemented directly with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import sigmoid


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler must be fitted before transform")
        return (values - self.mean_) / self.scale_

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)


@dataclass
class PrincipalComponentAnalysis:
    n_components: int = 2
    components_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    mean_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "PrincipalComponentAnalysis":
        if not 1 <= self.n_components <= values.shape[1]:
            raise ValueError("invalid number of components")
        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        covariance = centered.T @ centered / (len(values) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        self.components_ = eigenvectors[:, order[: self.n_components]].T
        total = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[: self.n_components] / total
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform")
        return (values - self.mean_) @ self.components_.T


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.08
    max_iterations: int = 2_000
    l2: float = 0.01
    tolerance: float = 1e-8
    weights_: np.ndarray | None = None
    bias_: float = 0.0
    loss_history_: list[float] = field(default_factory=list)

    def fit(self, values: np.ndarray, targets: np.ndarray) -> "LogisticRegressionGD":
        n_samples, n_features = values.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []
        previous = float("inf")

        for _ in range(self.max_iterations):
            probabilities = sigmoid(values @ self.weights_ + self.bias_)
            error = probabilities - targets
            weight_gradient = values.T @ error / n_samples + self.l2 * self.weights_
            bias_gradient = float(error.mean())
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient

            eps = 1e-12
            loss = -np.mean(
                targets * np.log(probabilities + eps)
                + (1 - targets) * np.log(1 - probabilities + eps)
            ) + 0.5 * self.l2 * float(self.weights_ @ self.weights_)
            self.loss_history_.append(float(loss))
            if abs(previous - loss) < self.tolerance:
                break
            previous = loss
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("model must be fitted before prediction")
        return sigmoid(values @ self.weights_ + self.bias_)

    def predict(self, values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(values) >= threshold).astype(int)


@dataclass
class KMeans:
    n_clusters: int = 3
    max_iterations: int = 200
    tolerance: float = 1e-5
    seed: int = 42
    centroids_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "KMeans":
        if not 1 < self.n_clusters <= len(values):
            raise ValueError("invalid number of clusters")
        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(len(values), self.n_clusters, replace=False)
        centroids = values[chosen].copy()
        for _ in range(self.max_iterations):
            labels = self._nearest(values, centroids)
            updated = np.vstack(
                [
                    values[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                    for i in range(self.n_clusters)
                ]
            )
            if np.linalg.norm(updated - centroids) <= self.tolerance:
                centroids = updated
                break
            centroids = updated
        self.centroids_ = centroids
        return self

    @staticmethod
    def _nearest(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        squared_distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return squared_distances.argmin(axis=1)

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("k-means must be fitted before prediction")
        return self._nearest(values, self.centroids_)


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    predictions = (probabilities >= threshold).astype(int)
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(targets),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(targets, probabilities),
        "decision_threshold": threshold,
    }


def select_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a threshold using training data only, avoiding test-set leakage."""
    candidates = np.linspace(0.1, 0.9, 161)
    scored = [classification_metrics(targets, probabilities, float(t))["f1"] for t in candidates]
    return float(candidates[int(np.argmax(scored))])


def roc_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("ROC-AUC requires both target classes")
    comparisons = positives[:, None] - negatives[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def roc_curve(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    positives = max(int(np.sum(targets == 1)), 1)
    negatives = max(int(np.sum(targets == 0)), 1)
    tpr, fpr = [], []
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(np.sum(predicted & (targets == 1)) / positives)
        fpr.append(np.sum(predicted & (targets == 0)) / negatives)
    return np.asarray(fpr), np.asarray(tpr)
