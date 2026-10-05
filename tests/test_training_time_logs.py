import csv
import tempfile
import unittest
from pathlib import Path

from utils.loss_curve_logger import LossCurveLogger
from utils.view_loss_curves import read_loss_rows


class TrainingTimeLogsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def logger(self, **kwargs):
        logger = LossCurveLogger(str(self.root), **kwargs)
        self.addCleanup(logger.close)
        return logger

    def test_resume_truncates_future_and_does_not_supply_clock_offset(self):
        logger = self.logger()
        for iteration, seconds in ((1, 1), (2, 2), (3, 3)):
            logger.log(iteration, 0.2, 0.3, training_time_seconds=seconds)
        logger.close()
        logger = self.logger(resume_from_iter=2, resume_training_seconds=2.5)
        self.assertEqual([row["iteration"] for row in logger.rows], [1, 2])
        logger.log(3, 0.2, 0.3, training_time_seconds=3.5)
        logger.close()
        rows = read_loss_rows(str(self.root / "loss_curves/losses.csv"))
        self.assertEqual([row["training_time_seconds"] for row in rows], [1, 2, 3.5])
        self.assertEqual(rows[-1]["training_time_scope"], "full_training")

    def test_checkpoint_time_wins_over_inconsistent_log_rows(self):
        logger = self.logger()
        logger.log(1, 0.2, 0.3, training_time_seconds=1)
        logger.log(2, 0.2, 0.3, training_time_seconds=500)
        logger.close()
        logger = self.logger(resume_from_iter=2, resume_training_seconds=2)
        self.assertEqual([row["iteration"] for row in logger.rows], [1])

    def test_resume_without_log_is_allowed(self):
        logger = self.logger(resume_from_iter=12, resume_training_seconds=42)
        self.assertEqual(logger.rows, [])
        logger.log(13, 0.2, 0.3, training_time_seconds=43)
        self.assertEqual(logger.rows[-1]["training_time_seconds"], 43)

    def test_legacy_logs_are_archived_without_mixing_time(self):
        output = self.root / "loss_curves"
        output.mkdir()
        old_csv = "iteration,image_loss,total_loss,training_time_seconds\n2,0.2,0.3,999\n"
        (output / "losses.csv").write_text(old_csv)
        (output / "training_time.html").write_text("legacy curve")
        logger = self.logger(
            resume_from_iter=2, training_time_scope="since_resume", legacy_resume=True,
        )
        self.assertEqual(logger.rows, [])
        archive = Path(logger.archive_path)
        self.assertEqual((archive / "losses.csv").read_text(), old_csv)
        self.assertEqual((archive / "training_time.html").read_text(), "legacy curve")
        logger.log(3, 0.2, 0.3, training_time_seconds=1)
        logger.close()
        self.assertIn("since_resume", (output / "training_time.html").read_text())
        with (output / "losses.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["iteration"], "3")

    def test_new_run_preserves_existing_outputs_in_unique_archives(self):
        logger = self.logger()
        logger.log(1, 0.2, 0.3, training_time_seconds=5)
        logger.close()
        fresh = self.logger()
        first_archive = fresh.archive_path
        self.assertIsNotNone(first_archive)
        self.assertEqual(fresh.rows, [])
        fresh.close()
        another = self.logger()
        self.assertNotEqual(first_archive, another.archive_path)
        self.assertTrue(Path(first_archive).is_dir())

    def test_close_failure_still_closes_file(self):
        logger = self.logger()
        logger.write_html = lambda: (_ for _ in ()).throw(OSError("HTML failure"))
        with self.assertRaises(OSError):
            logger.close()
        self.assertTrue(logger._csv_file.closed)
        logger.close()


if __name__ == "__main__":
    unittest.main()
