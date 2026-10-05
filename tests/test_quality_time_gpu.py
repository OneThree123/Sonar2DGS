"""Opt-in CUDA + Poisson quality test; all data/output is temporary."""

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


@unittest.skipUnless(os.environ.get("RUN_QUALITY_TIME_GPU_TESTS") == "1", "Opt-in CUDA/Poisson quality smoke test")
class QualityTimeGPUTest(unittest.TestCase):
    def test_quality_geometry_outputs_three_triggers_and_resume(self):
        import numpy as np
        import torch
        from PIL import Image

        import train
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from utils import geometry_metrics
        from utils.gaussian_mesh import write_ply_pointcloud
        from utils.quality_time_monitor import QualityTimeMonitor
        from utils.training_clock import TrainingClock

        self.addCleanup(train.network_gui.listener.close)
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory(prefix="quality-time-gpu-") as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            info = dict(img_width=16, img_height=32, tx_width=16, tx_height=16,
                        range_max=3.0, fovx=.7, fovy=.4)
            (dataset / "basic_info.json").write_text(json.dumps(info))
            for split, count in (("train", 2), ("test", 1)):
                (dataset / split).mkdir()
                frames = []
                for index in range(count):
                    relative = f"{split}/{index}.png"
                    Image.fromarray(np.arange(512, dtype=np.uint8).reshape(32, 16)).save(dataset / relative)
                    transform = np.eye(4)
                    transform[1, 3] = index * .1
                    frames.append(dict(file_path=relative, c2w=transform.tolist()))
                (dataset / split / "pose.json").write_text(json.dumps(frames))
            rng = np.random.default_rng(0)
            normals = rng.normal(size=(500, 3))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            points = normals * .15 + [1.5, 0, 0]
            for name in ("dense_points3d.ply", "points3d.ply"):
                write_ply_pointcloud(dataset / name, points, normals=normals)

            parser = ArgumentParser()
            model = ModelParams(parser)
            opt = OptimizationParams(parser)
            pipeline = PipelineParams(parser)
            args = parser.parse_args([])
            args.source_path = str(dataset)
            args.model_path = str(root / "output")
            args.initial_opacity = .5
            args.iterations = 4
            args.stop_after_training_seconds = 1e-6
            args.densify_from_iter = 0
            args.densify_until_iter = 10
            args.densification_interval = 2
            args.densify_grad_threshold = 1e6
            args.opacity_cull = 0
            args.opacity_reset_interval = 2
            # Retain geometry after resetting so the resumed mesh can be measured too.
            args.opacity_reset_value = .3
            clocks = []
            monitors = []
            reports = []
            reference_export = geometry_metrics.export_poisson_mesh_from_pointcloud_ply
            real_report = train.training_report

            def clock_factory(*a, **kw):
                clock = TrainingClock(*a, **kw)
                clocks.append(clock)
                return clock

            def monitor_factory(*a, **kw):
                monitor = QualityTimeMonitor(*a, **kw, targets=(1e-6, 2e-6, 7000))
                monitors.append(monitor)
                return monitor

            def report(*a, **kw):
                before = clocks[-1].snapshot_seconds()
                time.sleep(.03)
                result = real_report(*a, **kw)
                self.assertEqual(before, clocks[-1].snapshot_seconds())
                reports.append(result)
                return result

            with patch.object(train, "TrainingClock", clock_factory), patch.object(
                train, "QualityTimeMonitor", monitor_factory,
            ), patch.object(train, "training_report", report), patch.object(
                train.network_gui, "try_connect", lambda *args: None,
            ), patch.object(geometry_metrics, "POISSON_CONFIG", dict(geometry_metrics.POISSON_CONFIG, poisson_depth=5)), patch.object(
                geometry_metrics, "PREDICTED_CONFIG", dict(geometry_metrics.PREDICTED_CONFIG, poisson_depth=5),
            ), patch.object(geometry_metrics, "export_poisson_mesh_from_pointcloud_ply", wraps=reference_export) as reference_builder:
                def run(checkpoint=None):
                    with contextlib.redirect_stdout(io.StringIO()):
                        train.training(model.extract(args), opt.extract(args), pipeline.extract(args),
                                       [2, 4], [2, 4], [2, 4], checkpoint, 2, True, True)

                run()
                monitor = monitors[-1]
                self.assertEqual(reference_builder.call_count, 1)
                self.assertEqual(len(reports), 1)
                row = monitor.logger.rows[0]
                self.assertEqual(row["trigger_reasons"], ["time", "iteration", "final"])
                self.assertEqual(row["covered_target_seconds"], [1e-6, 2e-6])
                self.assertEqual(row["p2s_status"], "ok", row["error"])
                self.assertEqual(row["psnr_status"], "ok")
                self.assertEqual(row["p2s_unit"], "m")
                self.assertEqual(row["actual_training_seconds"], clocks[-1].finish())
                self.assertTrue(list((monitor.logger.output_dir / "evaluations").glob("*/renders/*.png")))
                self.assertTrue(Path(row["pred_mesh_path"]).is_file())
                self.assertFalse((root / "output/test_renders").exists())
                saved_mesh = root / "output/point_cloud/iteration_2/poisson_mesh.ply"
                self.assertEqual(saved_mesh.read_bytes(), Path(row["pred_mesh_path"]).read_bytes())

                checkpoint = root / "output/chkpnt2.pth"
                state = torch.load(checkpoint, weights_only=False)
                self.assertEqual(state["pending_opacity_reset"]["origin_group"], 1)
                run(str(checkpoint))  # Budget already reached: no tests and no reset.
                self.assertEqual(len(reports), 1)
                self.assertEqual(reference_builder.call_count, 1)
                args.stop_after_training_seconds = 0
                run(str(checkpoint))
                self.assertEqual(reference_builder.call_count, 1)
                self.assertEqual(len(reports), 2)
                self.assertEqual(monitors[-1].logger.rows[-1]["covered_target_seconds"], [])
                self.assertEqual(monitors[-1].logger.rows[-1]["trigger_reasons"], ["iteration", "final"])
                self.assertEqual(monitors[-1].logger.rows[-1]["p2s_status"], "ok")
                with monitors[-1].logger.csv_path.open() as output:
                    self.assertEqual(len(list(csv.DictReader(output))), 2)
                print("CUDA quality smoke: GT built once; PSNR + predicted Poisson + P2S succeeded; "
                      "triggers merged, time frozen, mesh reused and checkpoint resumed.")


if __name__ == "__main__":
    unittest.main()
