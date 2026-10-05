"""Exercise the actual report function with known images, without CUDA extensions."""

import ast
import contextlib
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from utils.image_utils import psnr
from utils.loss_utils import l1_loss


class QualityPSNRTest(unittest.TestCase):
    def report(self, values, **kwargs):
        source = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"training_report", "save_tensor_image"}]
        namespace = dict(torch=torch, os=os, math=math, Image=Image, np=np,
                         l1_loss=l1_loss, psnr=psnr,
                         normalize_render_intensity_for_output=lambda package, view: package)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        cameras = [SimpleNamespace(image_name=str(index), original_image=torch.zeros(1, 2, 2),
                                   prediction=value, anti_noise_image=torch.ones(1, 2, 2))
                   for index, value in enumerate(values)]
        scene = SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.zeros(1, 3)), model_path="unused",
                                getTestCameras=lambda: cameras, getTrainCameras=lambda: [])
        render = lambda view, *args: {"r_intensity": torch.full((1, 2, 2), view.prediction)}
        with contextlib.redirect_stdout(io.StringIO()):
            return namespace["training_report"](None, 2, [2], scene, render, (), **kwargs)

    def test_mean_of_per_view_psnr_and_original_gt(self):
        result = self.report([.1, .5], save_test_images=False, anti_noise_enabled=True)
        expected = (-20 * math.log10(.1) - 20 * math.log10(.5)) / 2
        self.assertAlmostEqual(result["test_psnr_db"], expected, places=5)
        self.assertEqual(result["num_test_views"], 2)
        self.assertEqual(result["psnr_status"], "ok")

    def test_empty_and_perfect_prediction_have_explicit_status(self):
        self.assertEqual(self.report([], save_test_images=False)["psnr_status"], "empty_test_set")
        result = self.report([0.], save_test_images=False)
        self.assertIsNone(result["test_psnr_db"])
        self.assertEqual(result["psnr_status"], "positive_infinity")

    def test_images_use_the_supplied_quality_directory(self):
        with tempfile.TemporaryDirectory() as root:
            self.report([.1], output_dir=root, save_test_images=True)
            self.assertEqual(len(list((Path(root) / "renders").glob("*.png"))), 1)
            self.assertEqual(len(list((Path(root) / "ground_truth").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
