import unittest

import numpy as np

from crm_intelligence.data import generate_customers, train_test_split
from crm_intelligence.models import (
    KMeans,
    LogisticRegressionGD,
    PrincipalComponentAnalysis,
    StandardScaler,
    classification_metrics,
    roc_auc,
    select_f1_threshold,
)


class ModelTests(unittest.TestCase):
    def test_scaler_centres_and_scales(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        transformed = StandardScaler().fit_transform(values)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-12)

    def test_pca_components_are_orthonormal(self):
        values = np.random.default_rng(4).normal(size=(100, 4))
        pca = PrincipalComponentAnalysis(2).fit(values)
        np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(2), atol=1e-10)
        self.assertEqual(pca.transform(values).shape, (100, 2))

    def test_auc_known_ordering(self):
        targets = np.array([0, 0, 1, 1])
        self.assertEqual(roc_auc(targets, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(roc_auc(targets, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_logistic_model_learns_signal(self):
        dataset = generate_customers(800, seed=7)
        train, test = train_test_split(dataset, seed=7)
        scaler = StandardScaler().fit(train.features)
        model = LogisticRegressionGD().fit(scaler.transform(train.features), train.churned)
        probabilities = model.predict_proba(scaler.transform(test.features))
        metrics = classification_metrics(test.churned, probabilities)
        self.assertGreater(metrics["roc_auc"], 0.7)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_threshold_is_selected_from_candidates(self):
        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = select_f1_threshold(targets, probabilities)
        self.assertGreaterEqual(threshold, 0.1)
        self.assertLessEqual(threshold, 0.9)
        self.assertEqual(classification_metrics(targets, probabilities, threshold)["f1"], 1.0)

    def test_kmeans_separates_simple_groups(self):
        values = np.array([[-5.1], [-4.9], [4.9], [5.1]])
        labels = KMeans(n_clusters=2, seed=2).fit(values).predict(values)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == "__main__":
    unittest.main()
