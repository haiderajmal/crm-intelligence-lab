import unittest

from crm_intelligence.experiment import bayesian_ab_test


class ExperimentTests(unittest.TestCase):
    def test_clear_treatment_improvement(self):
        result = bayesian_ab_test(100, 1_000, 140, 1_000, draws=30_000, seed=9)
        self.assertGreater(result.probability_treatment_better, 0.99)
        self.assertAlmostEqual(result.absolute_lift, 0.04)
        self.assertGreater(result.credible_interval_high, result.credible_interval_low)

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            bayesian_ab_test(11, 10, 3, 10)


if __name__ == "__main__":
    unittest.main()
