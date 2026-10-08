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
from utils.intensity_normalization import normalize_test_intensity
from utils.loss_utils import l1_loss


class QualityPSNRTest(unittest.TestCase):
    def report(self, values, *, gt_images=None, normalizing_methods=None,
               train_values=(), tb_writer=None, latest_batch_color_scale_mean=.25, **kwargs):
        source = Path(__file__).resolve().parents[1] / "train.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"training_report", "save_tensor_image"}]
        self.psnr_inputs = []
        self.l1_inputs = []
        self.legacy_views = []

        def metric(function, inputs):
            def evaluate(image, gt):
                inputs.append((image.clone(), gt.clone()))
                return function(image, gt)
            return evaluate

        namespace = dict(torch=torch, os=os, math=math, Image=Image, np=np,
                         l1_loss=metric(l1_loss, self.l1_inputs),
                         psnr=metric(psnr, self.psnr_inputs),
                         normalize_test_intensity=normalize_test_intensity)
        renderer_source = source.parent / "gaussian_renderer/__init__.py"
        renderer_tree = ast.parse(renderer_source.read_text())
        renderer_nodes = [node for node in renderer_tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"normalize_sonar_intensity", "normalize_render_intensity_for_output"}]
        exec(compile(ast.Module(body=renderer_nodes, type_ignores=[]), str(renderer_source), "exec"), namespace)
        legacy_normalize = namespace["normalize_render_intensity_for_output"]

        def normalize_legacy(package, view):
            self.legacy_views.append(view.image_name)
            return legacy_normalize(package, view)

        namespace["normalize_render_intensity_for_output"] = normalize_legacy
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)

        def frame(value):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            return tensor.expand(1, 2, 2).clone() if tensor.ndim == 0 else tensor.clone()

        cameras = [SimpleNamespace(
            image_name=str(index), prediction=frame(value),
            original_image=frame(gt_images[index]) if gt_images is not None else torch.zeros_like(frame(value)),
            normalizing_method=normalizing_methods[index] if normalizing_methods is not None else "all",
            anti_noise_image=torch.ones_like(frame(value)),
        ) for index, value in enumerate(values)]
        train_cameras = [SimpleNamespace(image_name=f"train_{index}", prediction=frame(value),
                                        original_image=torch.zeros_like(frame(value)), normalizing_method="range")
                         for index, value in enumerate(train_values)]
        scene = SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.zeros(1, 3)), model_path="unused",
                                getTestCameras=lambda: cameras, getTrainCameras=lambda: train_cameras)
        self.rendered_views = []

        def render(view, *args):
            self.rendered_views.append(view.image_name)
            return {"raw_intensity": view.prediction}
        with contextlib.redirect_stdout(io.StringIO()):
            return namespace["training_report"](tb_writer, 2, [2], scene, render, (),
                                                 latest_batch_color_scale_mean=latest_batch_color_scale_mean, **kwargs)

    def test_mean_of_per_view_psnr_and_original_gt(self):
        first = torch.tensor([[[0., 1.], [2., 4.]]])
        second = torch.tensor([[[0., 0.], [0., 2.]]])
        gt = torch.full_like(first, .2)
        result = self.report([first, second], gt_images=[gt, gt],
                             save_test_images=False, anti_noise_enabled=True)
        # Both test views use the same latest training batch multiplier (.25).
        images = [torch.tensor([[[0., .25], [.5, 1.]]]),
                  torch.tensor([[[0., 0.], [0., .5]]])]
        per_view = [-10 * math.log10(((image.double() - gt.double()) ** 2).mean().item()) for image in images]
        expected = sum(per_view) / 2
        self.assertAlmostEqual(result["test_psnr_db"], expected, places=5)
        self.assertEqual(result["num_test_views"], 2)
        self.assertEqual(result["psnr_status"], "ok")
        for (image, actual_gt), expected_image in zip(self.psnr_inputs, images):
            torch.testing.assert_close(image, expected_image)
            torch.testing.assert_close(actual_gt, gt)
        self.assertEqual(self.legacy_views, [])

    def test_png_tensorboard_l1_and_psnr_use_the_same_processed_image(self):
        raw = torch.tensor([[[0., 1.], [2., 4.]]])
        expected = torch.tensor([[[0., .25], [.5, 1.]]])
        images, scalars = {}, {}
        writer = SimpleNamespace(add_images=lambda tag, value, **kw: images.update({tag: value.clone()}),
                                 add_scalar=lambda tag, value, step: scalars.update({tag: float(value)}))
        with tempfile.TemporaryDirectory() as root:
            result = self.report([raw], output_dir=root, tb_writer=writer)
            with Image.open(Path(root) / "renders/00000_0.png") as saved:
                np.testing.assert_array_equal(np.asarray(saved), (expected[0] * 255).round().byte().numpy())
            torch.testing.assert_close(images["test_view_0/render"], expected[None])
            torch.testing.assert_close(self.psnr_inputs[0][0], expected)
            torch.testing.assert_close(self.l1_inputs[0][0], expected)
            self.assertAlmostEqual(scalars["test/loss_viewpoint/l1_loss"], expected.mean().item())
            self.assertEqual(scalars["test/loss_viewpoint/psnr"], result["test_psnr_db"])
            without_export = self.report([raw], save_test_images=False)
            self.assertEqual(result, without_export)
        torch.testing.assert_close(raw, torch.tensor([[[0., 1.], [2., 4.]]]))

    def test_camera_range_setting_does_not_change_test_normalization(self):
        raw = torch.tensor([[[1., 2.], [10., 20.]]])
        self.report([raw, raw], normalizing_methods=["all", "range"], save_test_images=False,
                    latest_batch_color_scale_mean=.01)
        torch.testing.assert_close(self.psnr_inputs[0][0], self.psnr_inputs[1][0])
        torch.testing.assert_close(self.psnr_inputs[0][0], raw * .01)
        self.assertEqual(self.legacy_views, [])

    def test_brightness_and_test_ground_truth_do_not_reestimate_scale(self):
        raw = torch.tensor([[[0., .1], [.2, .3]]])
        self.report([raw, raw * 2], gt_images=[torch.zeros_like(raw), torch.ones_like(raw)],
                    latest_batch_color_scale_mean=2., save_test_images=False)
        torch.testing.assert_close(self.psnr_inputs[0][0], raw * 2)
        torch.testing.assert_close(self.psnr_inputs[1][0], (raw * 4).clamp(0, 1))

    def test_missing_latest_batch_mean_rejects_test_rendering(self):
        with self.assertRaisesRegex(ValueError, "latest batch color scale mean"):
            self.report([.1], latest_batch_color_scale_mean=None, save_test_images=False)

    def test_train_diagnostics_keep_legacy_range_normalization(self):
        raw = torch.tensor([[[1., 2.], [10., 20.]]])
        self.report([raw], train_values=[raw], save_test_images=False)
        self.assertEqual(self.legacy_views, ["train_0"])
        torch.testing.assert_close(self.psnr_inputs[1][0], torch.tensor([[[.5, 1.], [.5, 1.]]]))

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

    def test_iteration_images_saved_when_quality_image_export_is_disabled(self):
        with tempfile.TemporaryDirectory() as root:
            quality = Path(root) / "quality"
            regular = Path(root) / "test_renders/iteration_2"
            self.report([.1, .5], output_dir=quality, save_test_images=False,
                        additional_image_output_dirs=(regular,), anti_noise_enabled=True)
            self.assertFalse(quality.exists())
            for name in ("renders", "ground_truth", "ground_truth_anti_noise"):
                self.assertEqual(len(list((regular / name).glob("*.png"))), 2)
            self.assertEqual(self.rendered_views, ["0", "1"])

    def test_two_destinations_share_one_render_and_identical_images(self):
        with tempfile.TemporaryDirectory() as root:
            quality = Path(root) / "quality"
            regular = Path(root) / "test_renders/iteration_2"
            self.report([.1, .5], output_dir=quality, save_test_images=True,
                        additional_image_output_dirs=(regular,), anti_noise_enabled=True)
            self.assertEqual(self.rendered_views, ["0", "1"])
            for image in regular.rglob("*.png"):
                self.assertEqual(image.read_bytes(), (quality / image.relative_to(regular)).read_bytes())


if __name__ == "__main__":
    unittest.main()
