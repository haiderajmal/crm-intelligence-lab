import tempfile
import unittest
from pathlib import Path

import numpy as np

from crm_intelligence.data import generate_customers
from crm_intelligence.database import CRMDatabase


class DatabaseTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            database = CRMDatabase(Path(directory) / "test.sqlite3")
            database.initialise()
            dataset = generate_customers(100, seed=3)
            database.replace_customers(dataset, np.full(100, 0.25), np.zeros(100, dtype=int))
            database.add_experiment({"probability_treatment_better": 0.98})
            self.assertEqual(database.customer_count(), 100)
            self.assertEqual(database.get_customer(1)["churn_probability"], 0.25)
            self.assertEqual(database.latest_experiment()["probability_treatment_better"], 0.98)


if __name__ == "__main__":
    unittest.main()
