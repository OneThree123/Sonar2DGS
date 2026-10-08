import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.quality_curve_logger import QualityCurveLogger
from utils.intensity_normalization import TEST_RENDER_NORMALIZATION_CONFIG


class QualityLogsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = dict(targets=[10, 20, 30], unit="m")

    def record(self, iteration, targets):
        return dict(iteration=iteration, group=iteration // 2, actual_training_seconds=iteration * 6,
                    covered_target_seconds=targets, trigger_reasons=["time", "iteration"],
                    test_psnr_db=20., p2s=None, psnr_status="ok", p2s_status="failed",
                    error="invalid mesh </script>")

    def test_each_evaluation_is_one_csv_row_and_embedded_json_is_safe(self):
        logger = QualityCurveLogger(self.root, self.config)
        logger.append(self.record(4, [10, 20]))
        logger.close(dict(remaining_targets=[30], termination_reason="iteration_limit"))
        with logger.csv_path.open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["covered_target_seconds"]), [10, 20])
        self.assertEqual(rows[0]["p2s"], "")
        html = logger.html_path.read_text()
        self.assertNotIn("invalid mesh </script>", html)
        self.assertIn("\\u003c/script>", html)
        self.assertNotIn("https://", html)
        self.assertTrue(logger._jsonl_file.closed)

    def test_checkpoint_rolls_back_future_results_and_incomplete_tail(self):
        logger = QualityCurveLogger(self.root, self.config)
        logger.append(self.record(2, [10]))
        state = dict(schema_version=1, run_id=logger.run_id,
                     config_signature=logger.config_signature, last_evaluation_id=1)
        logger.append(self.record(4, [20]))
        logger.close()
        with logger.jsonl_path.open("a") as file:
            file.write('{"partial":')
        resumed = QualityCurveLogger(self.root, self.config, state=state, resume_iteration=2)
        self.assertEqual([row["iteration"] for row in resumed.rows], [2])
        self.assertEqual(resumed.next_evaluation_id, 2)
        resumed.close()
        self.assertEqual(len(resumed.jsonl_path.read_text().splitlines()), 1)

    def test_missing_history_does_not_reset_committed_id(self):
        logger = QualityCurveLogger(self.root, self.config)
        state = dict(schema_version=1, run_id=logger.run_id,
                     config_signature=logger.config_signature, last_evaluation_id=2)
        logger.close()
        logger.jsonl_path.unlink()
        resumed = QualityCurveLogger(self.root, self.config, state=state, resume_iteration=4)
        self.assertEqual(resumed.next_evaluation_id, 3)
        self.assertEqual(resumed.missing_history, 2)
        resumed.close()

    def test_config_changes_are_rejected_before_log_mutation(self):
        logger = QualityCurveLogger(self.root, self.config)
        logger.append(self.record(2, [10]))
        state = dict(schema_version=1, run_id=logger.run_id,
                     config_signature=logger.config_signature, last_evaluation_id=1)
        logger.close()
        before = logger.jsonl_path.read_text()
        with self.assertRaises(ValueError):
            QualityCurveLogger(self.root, dict(self.config, unit="cm"), state=state)
        self.assertEqual(logger.jsonl_path.read_text(), before)

    def test_empty_log_and_nonfinite_values_remain_readable(self):
        logger = QualityCurveLogger(self.root, self.config)
        self.assertIn('"rows": []', logger.html_path.read_text())
        row = self.record(2, [])
        row["test_psnr_db"] = float("inf")
        row["psnr_status"] = "positive_infinity"
        logger.append(row)
        logger.close()
        record = json.loads(logger.jsonl_path.read_text())
        self.assertIsNone(record["test_psnr_db"])
        self.assertEqual(record["psnr_status"], "positive_infinity")

    def test_old_psnr_protocol_rejected_without_modifying_quality_artifacts(self):
        old_config = dict(self.config, psnr_protocol="mean_of_view_psnr_original_gt_normalized_clamped_render")
        logger = QualityCurveLogger(self.root, old_config)
        logger.append(self.record(2, [10]))
        state = dict(schema_version=1, run_id=logger.run_id,
                     config_signature=logger.config_signature, last_evaluation_id=1)
        logger.close()
        before = {path: path.read_bytes() for path in logger.output_dir.iterdir() if path.is_file()}
        with self.assertRaisesRegex(ValueError, "protocol does not match"):
            QualityCurveLogger(self.root, dict(self.config, **TEST_RENDER_NORMALIZATION_CONFIG), state=state)
        self.assertEqual(before, {path: path.read_bytes() for path in logger.output_dir.iterdir() if path.is_file()})


if __name__ == "__main__":
    unittest.main()
