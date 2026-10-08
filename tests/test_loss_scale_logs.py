import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.loss_scale_logger import LossScaleLogger


class LossScaleLogsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cameras = [SimpleNamespace(image_name="a", normalizing_method="all"),
                        SimpleNamespace(image_name="b", normalizing_method="range")]

    def rows(self):
        with (self.root / "loss_scales.csv").open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_range_coefficients_are_preserved_and_spread_is_visible(self):
        logger = LossScaleLogger(self.root, self.cameras, "mean_match")
        logger.log(1, 1, 2, 10., [[1.], [1., 100.]], [1, 0])
        row = self.rows()[0]
        self.assertEqual(float(row[logger.image_columns[0]]), 1.)
        self.assertEqual(json.loads(row[logger.image_columns[1]]), [1., 100.])
        self.assertEqual(json.loads(row["training_order"]), [1, 0])
        self.assertEqual(float(row["scale_max_min_ratio"]), 100.)
        self.assertEqual(float(row["scale_mean"]), 34.)
        self.assertEqual(logger.latest_batch_mean, 34.)
        self.assertAlmostEqual(float(row["scale_std"]), 46.66904755831214)

    def test_checkpoint_rollback_preserves_archive_and_replaces_future_rows(self):
        logger = LossScaleLogger(self.root, self.cameras, "mean_match")
        logger.log(1, 1, 2, 10., [[1.], [2.]], [0, 1])
        logger.log(2, 3, 4, 20., [[3.], [4.]], [1, 0])
        previous = logger.csv_path.read_bytes()
        resumed = LossScaleLogger(self.root, self.cameras, "mean_match", resume_from_iter=2)
        self.assertEqual(resumed.archive_path.read_bytes(), previous)
        self.assertEqual(resumed.latest_batch_mean, 1.5)
        resumed.log(2, 3, 4, 21., [[30.], [40.]], [0, 1])
        self.assertEqual([row["group"] for row in self.rows()], ["1", "2"])
        self.assertEqual(float(self.rows()[-1][resumed.image_columns[0]]), 30.)
        self.assertEqual(resumed.latest_batch_mean, 35.)

    def test_restart_or_changed_mode_does_not_mix_history(self):
        for mode, resume in [("mean_match", 0), ("normalized", 2)]:
            with self.subTest(mode=mode, resume=resume):
                logger = LossScaleLogger(self.root, self.cameras, "mean_match")
                logger.log(1, 1, 2, 10., [[1.], [2.]], [0, 1])
                restarted = LossScaleLogger(self.root, self.cameras, mode, resume)
                self.assertTrue(restarted.archive_path.exists())
                self.assertEqual(self.rows(), [])
                self.assertIsNone(restarted.latest_batch_mean)

    def test_nonfinite_values_and_zero_denominators_are_not_hidden(self):
        logger = LossScaleLogger(self.root, self.cameras, "mean_match")
        logger.log(1, 1, 2, 10., [[0.], [0.]], [0, 1])
        logger.log(2, 3, 4, 20., [[float("inf")], [2.]], [0, 1])
        rows = self.rows()
        self.assertEqual(rows[0]["scale_cv"], "")
        self.assertEqual(rows[0]["scale_max_min_ratio"], "")
        self.assertEqual(rows[1][logger.image_columns[0]], "inf")
        self.assertEqual(rows[1]["scale_mean"], "")


if __name__ == "__main__":
    unittest.main()
