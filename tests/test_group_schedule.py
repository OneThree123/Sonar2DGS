import unittest

from utils.group_schedule import GroupTrainingSchedule


class GroupTrainingScheduleTest(unittest.TestCase):
    def test_total_iterations_round_up_to_complete_group(self):
        schedule = GroupTrainingSchedule(group_size=300, requested_iterations=1_000)

        self.assertEqual(schedule.total_groups, 4)
        self.assertEqual(schedule.effective_iterations, 1_200)
        self.assertEqual(schedule.rounding_overhead, 200)

    def test_explicit_events_round_up_and_merge(self):
        schedule = GroupTrainingSchedule(group_size=300, requested_iterations=1_000)

        self.assertEqual(
            schedule.quantize_events([100, 200, 300, 500, 1_000, 1_100, -1]),
            {
                1: (100, 200, 300),
                2: (500,),
                4: (1_000,),
            },
        )

    def test_periodic_events_do_not_accumulate_rounding_drift(self):
        schedule = GroupTrainingSchedule(group_size=300, requested_iterations=2_000)

        self.assertEqual(
            schedule.periodic_events(500),
            {
                2: (500,),
                4: (1_000,),
                5: (1_500,),
                7: (2_000,),
            },
        )

    def test_periodic_events_in_one_group_are_deduplicated(self):
        schedule = GroupTrainingSchedule(group_size=300, requested_iterations=1_000)

        self.assertEqual(
            schedule.periodic_events(
                100,
                from_iteration=500,
                until_iteration=1_000,
                until_inclusive=False,
            ),
            {
                2: (600,),
                3: (700, 800, 900),
            },
        )

    def test_exclusive_until_does_not_exclude_training_end_before_it(self):
        schedule = GroupTrainingSchedule(group_size=300, requested_iterations=1_000)

        self.assertEqual(
            schedule.periodic_events(
                100,
                from_iteration=500,
                until_iteration=15_000,
                until_inclusive=False,
            )[4],
            (1_000,),
        )


if __name__ == "__main__":
    unittest.main()
