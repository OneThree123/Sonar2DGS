"""Exercise the real training function on CPU without importing CUDA extensions."""

import ast
import atexit
import contextlib
import csv
import io
import math
import os
import random
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.group_schedule import GroupTrainingSchedule
from utils.loss_curve_logger import LossCurveLogger
from utils.loss_utils import masked_l1_loss, weighted_mean
from utils.training_clock import (
    TIMING_POLICY_VERSION, TrainingClock, checkpoint_training_time, validate_seconds,
)
from utils.quality_time_schedule import OPACITY_RESET_POLICY_VERSION


class TrainingTimeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.seconds = 0.0
        self.pending_gpu = 0.0
        self.output_delay = 100.0
        self.clock = None
        self.events = []
        self.writers = []
        self.loggers = []
        self.raise_in_evaluation = None

    def advance(self, name, seconds, *, training):
        if self.clock is not None:
            self.assertEqual(self.clock._started is not None, training, name)
        self.events.append(name)
        self.seconds += seconds

    def synchronize(self, device):
        self.assertEqual(device, torch.device("cpu"))
        self.seconds += self.pending_gpu
        self.pending_gpu = 0

    def namespace(self):
        test = self

        class Gaussians:
            def __init__(self):
                self.get_xyz = torch.nn.Parameter(torch.ones(3, 3))
                self.max_radii2D = torch.zeros(3)
                self.optimizer = SimpleNamespace(
                    step=lambda: test.advance("optimizer", 1, training=True),
                    zero_grad=lambda **kw: test.advance("zero_grad", 1, training=True),
                )

            def training_setup(self, opt):
                pass

            def update_learning_rate(self, iteration):
                test.advance("learning_rate", 1, training=True)

            def add_densification_stats(self, *args):
                test.advance("densification_stats", 1, training=True)

            def enforce_parameter_bounds(self):
                test.advance("parameter_bounds", 1, training=True)

            def densify_and_prune(self, *args, **kwargs):
                test.advance("densify_prune", 7, training=True)
                with kwargs["diagnostic_context"]():
                    test.advance("densify_debug", test.output_delay, training=False)
                    test.pending_gpu += test.output_delay

            def reset_opacity(self, value):
                test.advance("opacity_reset", 3, training=True)

            def capture(self):
                test.advance("capture", test.output_delay, training=False)
                return "model_state"

            def restore(self, parameters, opt):
                test.assertEqual(parameters, "model_state")


        camera = SimpleNamespace(original_image=torch.ones(1, 2, 2), gt_alpha_mask=None, image_name="test")

        class Scene:
            def __init__(self, dataset, gaussians, shuffle):
                self.model_path = dataset.model_path
                self.cameras_extent = 1.0

            def getTrainCameras(self):
                return [camera, camera]

            def getTestCameras(self):
                return [camera]

            def save(self, iteration, geometry_artifacts=None):
                test.advance("model_geometry_export", test.output_delay, training=False)
                test.pending_gpu += test.output_delay

        class Logger(LossCurveLogger):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                test.loggers.append(self)

            def log(self, *args, **kwargs):
                test.advance("CSV_HTML", test.output_delay, training=False)
                return super().log(*args, **kwargs)

            def close(self):
                test.advance("logger_close", test.output_delay, training=False)
                return super().close()

        class Writer:
            def __init__(self):
                self.closed = False
                self.scalars = []
                test.writers.append(self)

            def add_scalar(self, *args):
                test.advance("tensorboard", test.output_delay, training=False)
                self.scalars.append(args)

            def flush(self):
                test.advance("writer_flush", test.output_delay, training=False)

            def close(self):
                self.closed = True

        class Progress:
            def __init__(self, **kwargs):
                test.assertNotIn("elapsed", kwargs["bar_format"])

            def update(self, *args):
                test.advance("progress", test.output_delay, training=False)

            def set_postfix(self, values):
                test.assertIn("Training time", values)

            def close(self):
                pass

        gui = SimpleNamespace(conn=None, set_sonar_defaults=lambda cam: None)

        def connect(items):
            test.advance("GUI_connect", test.output_delay, training=False)
            gui.conn = True

        def receive():
            test.advance("GUI_receive", test.output_delay, training=False)
            test.pending_gpu += test.output_delay
            return None, True, False, 1.0, "Intensity"

        gui.try_connect = connect
        gui.receive = receive
        gui.send = lambda *args: test.advance("GUI_send", test.output_delay, training=False)

        def render(camera, gaussians, pipe, **kwargs):
            test.advance("training_forward", 1, training=True)
            test.pending_gpu += 2  # Training device tail is counted at pause.
            image = gaussians.get_xyz.sum().sigmoid().expand(1, 2, 2)
            return dict(
                raw_intensity=image, screenspace_points=gaussians.get_xyz,
                gs_visibility=torch.ones(3, dtype=torch.bool), gs_raddi=torch.ones(3),
                rend_dist=image, rend_alpha=image, rend_normal=image.expand(3, 2, 2),
                surf_normal=image.expand(3, 2, 2), surf_normal_mask=torch.ones_like(image),
            )

        def report(*args, **kwargs):
            test.advance("evaluation_PSNR_P2S", test.output_delay, training=False)
            test.pending_gpu += test.output_delay
            if test.raise_in_evaluation:
                raise test.raise_in_evaluation

        def clock_factory(initial_seconds, **kwargs):
            test.clock = TrainingClock(initial_seconds, now=lambda: test.seconds, **kwargs)
            return test.clock

        names = {
            "training", "normalize_opacity_reset_mode", "print_group_event_schedule",
            "annealed_lowpass_variance", "scheduled_max_gaussians",
            "get_total_gaussians", "get_anti_noise_settings",
        }
        source = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(source.read_text())
        nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "OPACITY_RESET_MODE_ALIASES"
                for target in node.targets
            )
        ]
        namespace = dict(
            os=os, math=math, random=random, atexit=atexit, torch=torch,
            GaussianModel=Gaussians, Scene=Scene, tqdm=Progress,
            LossCurveLogger=Logger, TrainingClock=clock_factory,
            GroupTrainingSchedule=GroupTrainingSchedule,
            TIMING_POLICY_VERSION=TIMING_POLICY_VERSION,
            OPACITY_RESET_POLICY_VERSION=OPACITY_RESET_POLICY_VERSION,
            checkpoint_training_time=checkpoint_training_time, validate_seconds=validate_seconds,
            prepare_output_and_logger=lambda dataset: Writer(), network_gui=gui,
            render=render, training_report=report, masked_l1_loss=masked_l1_loss,
            weighted_mean=weighted_mean, masked_ssim=lambda *args: torch.tensor(0.5),
            prepare_sonar_intensity_for_loss=lambda image, *args, **kw: (image, None),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        return namespace

    def run_training(self, *, budget=20, checkpoint=None, iterations=6,
                     testing_iterations=None, quality_time_test=False):
        self.clock = None
        parser = ArgumentParser()
        model = ModelParams(parser)
        opt = OptimizationParams(parser)
        pipe = PipelineParams(parser)
        args = parser.parse_args([])
        args.model_path = str(self.root)
        args.iterations = iterations
        args.stop_after_training_seconds = budget
        args.densify_from_iter = 0
        args.densify_until_iter = 100
        args.densification_interval = 2
        args.opacity_reset_interval = 2
        namespace = self.namespace()
        output = io.StringIO()
        save = torch.save

        def delayed_save(state, path):
            self.advance("checkpoint_save", self.output_delay, training=False)
            save(state, path)

        with patch.object(torch.cuda, "synchronize", self.synchronize), patch.object(
            torch, "save", delayed_save,
        ), contextlib.redirect_stdout(output):
            namespace["training"](
                model.extract(args), opt.extract(args), pipe.extract(args),
                [2, 4, 6] if testing_iterations is None else testing_iterations,
                [2, 4, 6], [2, 4, 6], checkpoint, 2, quality_time_test,
            )
        return output.getvalue()

    def test_structure_updates_cross_budget_and_all_outputs_are_excluded(self):
        text = self.run_training()
        self.assertIn("termination_reason=training_budget", text)
        self.assertEqual(self.clock.finish(), 22)  # Reset is deferred until the next group's start.
        self.assertEqual(self.events.count("optimizer"), 2)
        state = torch.load(self.root / "chkpnt2.pth", weights_only=False)
        self.assertEqual(state["training_time_seconds"], 22)
        self.assertEqual(state["pending_opacity_reset"]["origin_group"], 1)
        self.assertNotIn("opacity_reset", self.events)
        self.assertEqual(state["version"], 3)
        self.assertEqual(state["completed_groups"], 1)
        self.assertIn("capture", self.events)
        self.assertTrue(self.writers[-1].closed)
        time_scalars = [item[1] for item in self.writers[-1].scalars if item[0] == "train/training_time_seconds"]
        self.assertEqual(time_scalars, [8, 22])

    def test_evaluation_gui_and_output_delays_do_not_change_training_time(self):
        self.run_training(budget=0)
        self.assertEqual(self.clock.finish(), 72)
        self.assertIn("GUI_receive", self.events)
        self.output_delay = 10000
        self.run_training(budget=0)
        self.assertEqual(self.clock.finish(), 72)

    def test_resume_uses_checkpoint_and_skips_already_reached_budget(self):
        self.run_training()
        checkpoint = str(self.root / "chkpnt2.pth")
        self.events.clear()
        text = self.run_training(checkpoint=checkpoint)
        self.assertEqual(self.clock.finish(), 22)
        self.assertNotIn("optimizer", self.events)
        self.assertIn("no new group", text)
        self.run_training(budget=40, checkpoint=checkpoint)
        self.assertEqual(self.clock.finish(), 47)
        state = torch.load(self.root / "chkpnt4.pth", weights_only=False)
        self.assertEqual(state["training_time_seconds"], 47)
        with (self.root / "loss_curves/losses.csv").open() as file:
            self.assertEqual([int(row["iteration"]) for row in csv.DictReader(file)], [1, 2, 3, 4])

    def test_legacy_checkpoint_restarts_segment_and_archives_old_log(self):
        self.run_training()
        legacy = self.root / "legacy.pth"
        torch.save(("model_state", 2), legacy)
        self.run_training(checkpoint=str(legacy))
        self.assertEqual(self.clock.finish(), 22)
        self.assertIsNotNone(self.loggers[-1].archive_path)
        state = torch.load(self.root / "chkpnt4.pth", weights_only=False)
        self.assertEqual(state["training_time_scope"], "since_resume")

    def test_version2_checkpoint_can_restore_model_without_time_history(self):
        self.run_training()
        state = torch.load(self.root / "chkpnt2.pth", weights_only=False)
        state["version"] = 2
        for key in ("training_time_seconds", "timing_policy_version", "training_time_scope"):
            del state[key]
        state.pop("pending_opacity_reset")
        state.pop("opacity_reset_policy_version")
        checkpoint = self.root / "version2.pth"
        torch.save(state, checkpoint)
        self.run_training(checkpoint=str(checkpoint))
        self.assertEqual(self.clock.finish(), 22)
        state = torch.load(self.root / "chkpnt4.pth", weights_only=False)
        self.assertEqual(state["training_time_scope"], "since_resume")

    def test_invalid_budgets_fail_before_training_initialization(self):
        for budget in (-1, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.run_training(budget=budget)
        self.assertEqual(self.writers, [])
        self.assertNotIn("optimizer", self.events)

    def test_failure_and_interrupt_freeze_clock_and_close_outputs(self):
        for exception in (RuntimeError("evaluation failed"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                self.raise_in_evaluation = exception
                with self.assertRaises(type(exception)):
                    self.run_training()
                self.assertEqual(self.clock.finish(), 22)
                self.assertTrue(self.writers[-1].closed)
                self.assertTrue(self.loggers[-1]._csv_file.closed)


if __name__ == "__main__":
    unittest.main()
