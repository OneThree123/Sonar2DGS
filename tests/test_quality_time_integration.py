"""Real training loop and monitoring transactions, with controlled training/output clocks."""

import json
import csv
import time
from unittest.mock import patch

import torch

import test_training_time_integration as base
from utils.quality_time_monitor import QualityTimeMonitor
from utils.intensity_normalization import TEST_RENDER_NORMALIZATION_CONFIG


class QualityTimeIntegrationTest(base.TrainingTimeIntegrationTest):
    def namespace(self):
        namespace = super().namespace()
        test = self

        def monitor_factory(*args, **kwargs):
            class Geometry:
                def __init__(self, *args):
                    test.advance("reference_setup", test.reference_delay, training=False)
                    self.setup_seconds = test.reference_delay
                    self.config = dict(reference_signature="fake-fixed-reference", unit="m")

                def evaluate(self, gaussians, output_dir):
                    test.advance("predicted_poisson_P2S", test.output_delay, training=False)
                    return dict(p2s=0.2, p2s_unit="m", p2s_status="ok", geometry_status="ok", error=None), None

            with patch("utils.quality_time_monitor.GeometryEvaluator", Geometry):
                if test.quality_targets is not None:
                    kwargs["targets"] = test.quality_targets
                monitor = QualityTimeMonitor(*args, **kwargs)
            test.monitors.append(monitor)
            return monitor

        report = namespace["training_report"]

        def quality_report(*args, **kwargs):
            report(*args, **kwargs)
            return dict(test_psnr_db=25., num_test_views=1, psnr_status="ok",
                        latest_batch_color_scale_mean=kwargs["latest_batch_color_scale_mean"])

        namespace.update(QualityTimeMonitor=monitor_factory, training_report=quality_report)
        return namespace

    def setUp(self):
        super().setUp()
        self.monitors = []
        self.reference_delay = 100
        self.quality_targets = None
        self.timer_patch = patch.object(time, "perf_counter", lambda: self.seconds)
        self.timer_patch.start()
        self.addCleanup(self.timer_patch.stop)

    def test_three_triggers_and_multiple_targets_evaluate_once(self):
        self.run_training(iterations=2, budget=0, quality_time_test=True)
        monitor = self.monitors[-1]
        self.assertEqual(self.events.count("evaluation_PSNR_P2S"), 1)
        self.assertEqual(self.events.count("predicted_poisson_P2S"), 1)
        self.assertEqual(monitor.logger.rows[0]["covered_target_seconds"], [10, 20])
        self.assertEqual(monitor.logger.rows[0]["trigger_reasons"], ["time", "iteration", "final"])
        self.assertEqual(monitor.logger.rows[0]["actual_training_seconds"], 22)
        self.assertNotIn("opacity_reset", self.events)
        for key, value in TEST_RENDER_NORMALIZATION_CONFIG.items():
            self.assertEqual(monitor.logger.config[key], value)
        saved_config = json.loads((monitor.logger.output_dir / "config.json").read_text())
        self.assertEqual(saved_config["render_normalization"], "latest_batch_color_scale_mean")
        self.assertEqual(monitor.logger.rows[0]["latest_batch_color_scale_mean"], 1.)
        with monitor.logger.csv_path.open() as output:
            self.assertEqual(float(next(csv.DictReader(output))["latest_batch_color_scale_mean"]), 1.)

    def test_final_has_no_future_node_coverage_and_reset_is_after_evaluation(self):
        self.run_training(budget=0, iterations=6, testing_iterations=[], quality_time_test=True)
        rows = self.monitors[-1].logger.rows
        self.assertEqual([row["covered_target_seconds"] for row in rows], [[10, 20], [30], [50]])
        self.assertEqual(rows[-1]["trigger_reasons"], ["time", "final"])
        self.assertEqual(self.events.count("opacity_reset"), 2)
        self.assertLess(self.events.index("evaluation_PSNR_P2S"), self.events.index("opacity_reset"))
        summary = json.loads((self.monitors[-1].logger.output_dir / "summary.json").read_text())
        self.assertEqual(summary["remaining_targets"][0], 75)
        self.assertEqual(summary["pending_opacity_reset"]["origin_group"], 3)

    def test_reference_and_evaluation_delays_excluded_from_training(self):
        self.run_training(iterations=2, budget=0, quality_time_test=True)
        first = self.monitors[-1].logger.rows[0]
        self.output_delay = 10000
        self.run_training(iterations=2, budget=0, quality_time_test=True)
        second = self.monitors[-1].logger.rows[0]
        self.assertEqual(first["actual_training_seconds"], second["actual_training_seconds"])
        self.assertGreater(second["wall_seconds"], first["wall_seconds"])
        self.assertGreater(second["evaluation_seconds"], first["evaluation_seconds"])
        summary = json.loads((self.monitors[-1].logger.output_dir / "summary.json").read_text())
        self.reference_delay = 1000000
        self.run_training(iterations=2, budget=0, quality_time_test=True)
        changed_reference = json.loads((self.monitors[-1].logger.output_dir / "summary.json").read_text())
        self.assertEqual(summary["total_run_seconds"], changed_reference["total_run_seconds"])
        self.assertEqual(summary["final_training_seconds"], changed_reference["final_training_seconds"])

    def test_iteration_and_final_without_time_nodes_do_not_consume_future_targets(self):
        self.quality_targets = (1000, 2000)
        self.run_training(budget=0, iterations=6, testing_iterations=[4], quality_time_test=True)
        monitor = self.monitors[-1]
        self.assertEqual([r["iteration"] for r in monitor.logger.rows], [4, 6])
        self.assertEqual([r["trigger_reasons"] for r in monitor.logger.rows], [["iteration"], ["final"]])
        self.assertEqual([r["covered_target_seconds"] for r in monitor.logger.rows], [[], []])
        self.assertEqual(monitor.schedule.remaining, [1000, 2000])

    def test_restore_pending_reset_and_quality_nodes_once(self):
        self.run_training(quality_time_test=True)
        checkpoint = str(self.root / "chkpnt2.pth")
        state = torch.load(checkpoint, weights_only=False)
        self.assertEqual(state["quality_time_test_state"]["consumed_targets"], [10, 20])
        self.events.clear()
        self.run_training(checkpoint=checkpoint, quality_time_test=True)
        self.assertNotIn("opacity_reset", self.events)
        self.assertNotIn("evaluation_PSNR_P2S", self.events)
        self.events.clear()
        self.run_training(checkpoint=checkpoint, budget=40, quality_time_test=True)
        monitor = self.monitors[-1]
        self.assertEqual(self.events.count("opacity_reset"), 1)
        self.assertEqual([r["iteration"] for r in monitor.logger.rows], [2, 4])
        self.assertEqual(monitor.logger.rows[-1]["covered_target_seconds"], [30])

    def test_failure_closes_quality_history_without_consuming_nodes(self):
        self.raise_in_evaluation = RuntimeError("evaluation failed")
        with self.assertRaises(RuntimeError):
            self.run_training(quality_time_test=True)
        monitor = self.monitors[-1]
        self.assertTrue(monitor.logger._jsonl_file.closed)
        self.assertEqual(monitor.schedule.consumed, set())
        self.assertEqual(json.loads((monitor.logger.output_dir / "summary.json").read_text())["termination_reason"], "error")

    def test_iteration_exports_independent_of_quality_and_eval_flags(self):
        self.quality_targets = (1000,)
        for quality in (False, True):
            for images in (False, True):
                for eval_mode in (False, True):
                    with self.subTest(quality=quality, images=images, eval_mode=eval_mode):
                        self.report_calls.clear()
                        self.save_calls.clear()
                        self.run_training(budget=0, testing_iterations=[1, 2, 3],
                                          saving_iterations=[6], quality_time_test=quality,
                                          quality_save_test_images=images, eval_mode=eval_mode)
                        self.assertEqual(self.save_calls, [(2, False), (4, False), (6, True)])
                        for iteration, kwargs in self.report_calls:
                            if quality:
                                expected = (str(self.root / f"test_renders/iteration_{iteration}"),)
                                self.assertEqual(kwargs["additional_image_output_dirs"],
                                                 expected if iteration in (2, 4) else ())
                                self.assertEqual(kwargs["save_test_images"], images)
                            else:
                                self.assertNotIn("output_dir", kwargs)
                                self.assertTrue(kwargs.get("save_test_images", True))
                        self.assertEqual(self.clock.finish(), 72)

    def test_time_and_final_only_nodes_do_not_export_regular_images_or_gs(self):
        self.run_training(budget=0, testing_iterations=[4], saving_iterations=[],
                          quality_time_test=True)
        self.assertEqual(self.save_calls, [(4, False)])
        for iteration, kwargs in self.report_calls:
            self.assertFalse(kwargs["save_test_images"])
            self.assertEqual(kwargs["additional_image_output_dirs"],
                             (str(self.root / "test_renders/iteration_4"),) if iteration == 4 else ())

    def test_exports_preserve_parameter_updates_and_training_clock(self):
        original_namespace = self.namespace
        parameter_steps = []

        def namespace_with_parameter_updates():
            namespace = original_namespace()
            model_class = namespace["GaussianModel"]

            def model_factory():
                model = model_class()
                step = model.optimizer.step

                def update():
                    step()
                    with torch.no_grad():
                        model.get_xyz.add_(model.get_xyz.grad, alpha=-.01)
                    parameter_steps.append(model.get_xyz.detach().clone())

                model.optimizer.step = update
                return model

            namespace["GaussianModel"] = model_factory
            return namespace

        with patch.object(self, "namespace", namespace_with_parameter_updates):
            self.run_training(budget=0, testing_iterations=[], saving_iterations=[])
            baseline = torch.stack(parameter_steps)
            baseline_seconds = self.clock.finish()
            for quality in (False, True):
                for images in (False, True):
                    with self.subTest(quality=quality, images=images):
                        parameter_steps.clear()
                        self.run_training(budget=0, testing_iterations=[2, 4, 6],
                                          saving_iterations=[6], quality_time_test=quality,
                                          quality_save_test_images=images)
                        torch.testing.assert_close(torch.stack(parameter_steps), baseline,
                                                   rtol=0, atol=0)
                        self.assertEqual(self.clock.finish(), baseline_seconds)
