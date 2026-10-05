import unittest

from utils.training_clock import (
    TIMING_POLICY_VERSION, TrainingClock, checkpoint_training_time, validate_seconds,
)


class FakeTime:
    def __init__(self):
        self.seconds = 0.0
        self.pending_gpu = 0.0
        self.sync_calls = 0

    def now(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds

    def synchronize(self):
        self.sync_calls += 1
        self.advance(self.pending_gpu)
        self.pending_gpu = 0.0


class TrainingClockTest(unittest.TestCase):
    def setUp(self):
        self.time = FakeTime()
        self.clock = TrainingClock(now=self.time.now, synchronize=self.time.synchronize)

    def test_only_training_regions_accumulate_with_cpu_gpu_overlap(self):
        self.time.advance(20)  # Initialization.
        with self.clock.training_region():
            self.time.advance(2)  # CPU and GPU overlap in these two seconds.
            self.time.pending_gpu = 3  # Unfinished device work at the boundary.
        self.time.advance(100)  # Output and evaluation.
        with self.clock.training_region():
            self.time.advance(4)
        self.assertEqual(self.clock.finish(), 9)

    def test_nontraining_gpu_wait_is_excluded(self):
        self.time.pending_gpu = 50  # Initialization CUDA work.
        with self.clock.training_region():
            self.time.advance(2)
            self.time.pending_gpu = 3
            with self.clock.excluded_region():
                self.assertEqual(self.clock.snapshot_seconds(), 5)
                self.time.advance(40)
                self.time.pending_gpu = 60  # Evaluation CUDA tail.
            self.time.advance(4)
        self.assertEqual(self.clock.snapshot_seconds(), 9)
        self.assertEqual(self.time.seconds, 159)

    def test_nested_exclusions_restore_only_outer_state(self):
        with self.clock.training_region():
            self.time.advance(1)
            with self.clock.excluded_region(gpu_work=False):
                self.time.advance(10)
                with self.clock.excluded_region():
                    self.time.advance(20)
                    self.time.pending_gpu = 30
                self.assertEqual(self.clock.snapshot_seconds(), 1)
            self.time.advance(2)
        self.assertEqual(self.clock.finish(), 3)
        self.assertEqual(self.time.sync_calls, 4)

    def test_paused_exclusion_and_snapshots_never_resume(self):
        with self.clock.excluded_region():
            with self.clock.excluded_region():
                self.time.advance(10)
                self.time.pending_gpu = 20
        self.assertEqual(self.clock.snapshot_seconds(), 0)
        self.assertEqual(self.clock.snapshot_seconds(), 0)
        self.assertEqual(self.time.sync_calls, 0)
        with self.clock.training_region():
            self.time.advance(2)
        self.assertEqual(self.clock.finish(), 2)

    def test_cpu_only_output_does_not_add_resume_synchronization(self):
        with self.clock.training_region():
            self.time.advance(1)
            with self.clock.excluded_region(gpu_work=False):
                self.time.advance(50)
            self.time.advance(2)
        self.assertEqual(self.clock.finish(), 3)
        self.assertEqual(self.time.sync_calls, 3)  # Initial, pause, final pause.

    def test_nested_training_and_initial_offset(self):
        self.clock = TrainingClock(12, now=self.time.now)
        with self.clock.training_region():
            self.time.advance(1)
            with self.clock.training_region():
                self.time.advance(2)
            self.time.advance(3)
        self.assertEqual(self.clock.finish(), 18)

    def test_exception_unwinds_and_finish_is_idempotent(self):
        with self.assertRaisesRegex(ValueError, "test"):
            with self.clock.training_region():
                self.time.advance(2)
                with self.clock.excluded_region():
                    self.time.advance(100)
                    raise ValueError("test")
        self.assertEqual(self.clock.finish(), 2)
        self.time.advance(50)
        self.assertEqual(self.clock.finish(), 2)
        with self.assertRaises(RuntimeError):
            with self.clock.training_region():
                pass

    def test_finish_inside_exclusion_cannot_restart(self):
        with self.clock.training_region():
            self.time.advance(2)
            with self.clock.excluded_region():
                self.clock.finish()
                self.time.advance(50)
            self.time.advance(20)
        self.assertEqual(self.clock.finish(), 2)

    def test_running_snapshot_and_training_in_exclusion_are_rejected(self):
        with self.clock.training_region():
            with self.assertRaises(RuntimeError):
                self.clock.snapshot_seconds()
            with self.clock.excluded_region():
                with self.assertRaises(RuntimeError):
                    with self.clock.training_region():
                        pass

    def test_invalid_seconds_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_seconds(value)

    def test_device_failure_does_not_leave_clock_running(self):
        def failure():
            self.time.advance(3)
            raise RuntimeError("device failure")

        with self.assertRaisesRegex(RuntimeError, "device failure"):
            with self.clock.training_region():
                self.time.advance(2)
                self.clock._synchronize = failure
        self.assertEqual(self.clock.snapshot_seconds(), 5)
        self.assertEqual(self.clock.finish(), 5)


class CheckpointTrainingTimeTest(unittest.TestCase):
    def state(self):
        return dict(
            version=3, training_time_seconds=42.5,
            timing_policy_version=TIMING_POLICY_VERSION,
            training_time_scope="full_training",
        )

    def test_new_and_legacy_origins(self):
        self.assertEqual(checkpoint_training_time(None), (0, "full_training"))
        self.assertEqual(checkpoint_training_time(self.state()), (42.5, "full_training"))
        self.assertEqual(checkpoint_training_time({"version": 2}), (0, "since_resume"))
        self.assertEqual(checkpoint_training_time(("parameters", 12)), (0, "since_resume"))
        state = self.state()
        state["training_time_scope"] = "since_resume"
        self.assertEqual(checkpoint_training_time(state), (42.5, "since_resume"))

    def test_unknown_versions_scopes_and_invalid_time_are_rejected(self):
        for key, value in (
            ("version", 4), ("timing_policy_version", 99),
            ("training_time_scope", "unknown"), ("training_time_seconds", float("nan")),
        ):
            state = self.state()
            state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_training_time(state)


if __name__ == "__main__":
    unittest.main()
