import unittest

from utils.quality_time_schedule import QUALITY_TEST_SECONDS, QualityTimeSchedule


class QualityTimeScheduleTest(unittest.TestCase):
    def test_fixed_nodes_and_equality_boundary(self):
        schedule = QualityTimeSchedule()
        self.assertEqual(len(QUALITY_TEST_SECONDS), 26)
        self.assertEqual(schedule.due(9.999), [])
        self.assertEqual(schedule.triggers(10), (["time"], [10]))
        self.assertEqual(schedule.due(10), [10])  # Merely querying cannot consume a result.
        schedule.commit([10])
        self.assertEqual(schedule.due(10), [])

    def test_crossed_nodes_merge_with_iteration_and_final(self):
        schedule = QualityTimeSchedule(consumed=[10])
        self.assertEqual(schedule.triggers(54, iteration=True, final=True),
                         (["time", "iteration", "final"], [20, 30, 50]))
        schedule.commit([20, 30, 50])
        self.assertEqual(schedule.triggers(60, final=True), (["final"], []))
        self.assertEqual(schedule.remaining[0], 75)

    def test_final_does_not_consume_future_nodes(self):
        schedule = QualityTimeSchedule()
        self.assertEqual(schedule.triggers(9, final=True), (["final"], []))
        self.assertEqual(len(schedule.remaining), 26)

    def test_restore_by_target_value_and_historical_misses(self):
        schedule = QualityTimeSchedule(consumed=[10, 20], missed=[30])
        self.assertEqual(schedule.due(54), [50])
        self.assertEqual(schedule.state()["missed_before_resume_targets"], [30])
        with self.assertRaises(ValueError):
            schedule.commit([30])

    def test_invalid_times_and_state_are_rejected(self):
        for nodes in ([], [0], [-1], [float("inf")], [float("nan")]):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                QualityTimeSchedule(nodes)
        with self.assertRaises(ValueError):
            QualityTimeSchedule(consumed=[11])
        with self.assertRaises(ValueError):
            QualityTimeSchedule(consumed=[10], missed=[10])


if __name__ == "__main__":
    unittest.main()
