"""Opt-in smoke test: RUN_TRAINING_TIME_GPU_TESTS=1 in the training environment."""

import contextlib
import csv
import io
import json
import os
import tempfile
import time
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("RUN_TRAINING_TIME_GPU_TESTS") == "1", "Opt-in CUDA training smoke test")
class TrainingTimeGPUTest(unittest.TestCase):
    def test_training_debug_outputs_budget_and_checkpoint_resume(self):
        import numpy as np
        import torch
        from PIL import Image

        import train
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from utils.loss_curve_logger import LossCurveLogger
        from utils.training_clock import TrainingClock

        self.addCleanup(train.network_gui.listener.close)
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory(prefix="training-time-gpu-") as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            info = dict(
                img_width=16, img_height=32, tx_width=16, tx_height=16,
                range_max=3.0, fovx=0.7, fovy=0.4,
            )
            (dataset / "basic_info.json").write_text(json.dumps(info))
            image = np.arange(32 * 16, dtype=np.uint8).reshape(32, 16)
            for split, count in (("train", 2), ("test", 1)):
                (dataset / split).mkdir()
                frames = []
                for index in range(count):
                    relative = f"{split}/{index}.png"
                    Image.fromarray(image).save(dataset / relative)
                    transform = np.eye(4)
                    transform[1, 3] = index * 0.1
                    frames.append(dict(file_path=relative, c2w=transform.tolist()))
                (dataset / split / "pose.json").write_text(json.dumps(frames))

            parser = ArgumentParser()
            model = ModelParams(parser)
            optimizer = OptimizationParams(parser)
            pipeline = PipelineParams(parser)
            args = parser.parse_args([])
            args.source_path = str(dataset)
            args.model_path = str(root / "output")
            args.point_cloud_init = "sonar_frustum_random"
            args.point_cloud_init_count = 32
            args.iterations = 4
            args.stop_after_training_seconds = 1e-6
            args.densify_from_iter = 0
            args.densify_until_iter = 10
            args.densification_interval = 2
            args.densify_grad_threshold = 1e6  # Keep the smoke model small.
            args.densify_debug = True
            args.opacity_cull = 0
            args.opacity_reset_interval = 2

            clocks = []
            excluded_callbacks = []
            real_report = train.training_report
            real_save = train.Scene.save
            real_capture = train.GaussianModel.capture
            real_debug = train.GaussianModel._print_densify_debug
            real_html = LossCurveLogger.write_html

            def clock_factory(*a, **kw):
                clock = TrainingClock(*a, **kw)
                clocks.append(clock)
                return clock

            def excluded(name, callback, *a, **kw):
                before = clocks[-1].snapshot_seconds() if clocks else None
                time.sleep(0.04)
                # Queue asynchronous GPU work so the next training region must wait.
                torch.ones(16, device="cuda").sin()
                result = callback(*a, **kw)
                if before is not None:
                    self.assertEqual(before, clocks[-1].snapshot_seconds(), name)
                    excluded_callbacks.append(name)
                return result

            def run(checkpoint=None):
                torch.manual_seed(0)
                np.random.seed(0)
                text = io.StringIO()
                with contextlib.redirect_stdout(text), patch.object(train, "TrainingClock", clock_factory), patch.object(
                    train, "training_report", lambda *a, **kw: excluded("evaluation", real_report, *a, **kw),
                ), patch.object(train.Scene, "save", lambda *a, **kw: excluded("geometry_save", real_save, *a, **kw)), patch.object(
                    train.GaussianModel, "capture", lambda *a, **kw: excluded("capture", real_capture, *a, **kw),
                ), patch.object(train.GaussianModel, "_print_densify_debug", lambda *a, **kw: excluded("debug", real_debug, *a, **kw)), patch.object(
                    LossCurveLogger, "write_html", lambda *a, **kw: excluded("HTML", real_html, *a, **kw),
                ), patch.object(train.network_gui, "try_connect", lambda items: None):
                    train.training(
                        model.extract(args), optimizer.extract(args), pipeline.extract(args),
                        [2, 4], [2, 4], [2, 4], checkpoint, 2,
                    )
                return text.getvalue()

            text = run()
            output = root / "output"
            checkpoint2 = output / "chkpnt2.pth"
            state2 = torch.load(checkpoint2, weights_only=False)
            self.assertEqual(state2["version"], 3)
            self.assertEqual(state2["global_step"], 2)
            self.assertEqual(state2["completed_groups"], 1)
            self.assertEqual(state2["training_time_seconds"], clocks[-1].finish())
            self.assertIn("termination_reason=training_budget", text)
            self.assertFalse((output / "chkpnt4.pth").exists())
            for name in ("evaluation", "geometry_save", "capture", "debug", "HTML"):
                self.assertIn(name, excluded_callbacks)
            self.assertTrue((output / "point_cloud/iteration_2/point_cloud.ply").is_file())
            self.assertTrue(list((output / "test_renders/iteration_2/renders").glob("*.png")))

            text = run(str(checkpoint2))
            self.assertEqual(clocks[-1].finish(), state2["training_time_seconds"])
            self.assertIn("no new group", text)
            self.assertFalse((output / "chkpnt4.pth").exists())

            args.stop_after_training_seconds = 0
            text = run(str(checkpoint2))
            state4 = torch.load(output / "chkpnt4.pth", weights_only=False)
            self.assertGreater(state4["training_time_seconds"], state2["training_time_seconds"])
            self.assertEqual(state4["training_time_seconds"], clocks[-1].finish())
            self.assertIn("termination_reason=iteration_limit", text)
            with (output / "loss_curves/losses.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["iteration"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual(float(rows[-1]["training_time_seconds"]), state4["training_time_seconds"])
            print(
                f"CUDA smoke: group 1={state2['training_time_seconds']:.6f}s, "
                f"resumed total={state4['training_time_seconds']:.6f}s; "
                "evaluation/export/capture/debug/HTML delays excluded."
            )


if __name__ == "__main__":
    unittest.main()
