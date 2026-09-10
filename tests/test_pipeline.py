import json
import tempfile
import unittest
from pathlib import Path

from crm_intelligence.database import CRMDatabase
from crm_intelligence.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_pipeline(directory, customers=300, seed=42)
            output = Path(directory)
            for name in (
                "crm_lab.sqlite3",
                "metrics.json",
                "loss_curve.png",
                "roc_curve.png",
                "run_summary.md",
            ):
                self.assertTrue((output / name).is_file(), name)
            saved = json.loads((output / "metrics.json").read_text())
            self.assertEqual(saved["customers"], 300)
            self.assertEqual(CRMDatabase(output / "crm_lab.sqlite3").customer_count(), 300)
            self.assertGreater(result["model"]["roc_auc"], 0.65)


if __name__ == "__main__":
    unittest.main()
