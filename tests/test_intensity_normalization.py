"""Check the shared brightness protocol, including its training gradients."""

import unittest
from types import SimpleNamespace

import torch

from utils.intensity_utils import (
    normalize_render_intensity_for_output,
    normalize_sonar_intensity,
    prepare_sonar_intensity_for_loss,
)


class IntensityNormalizationTest(unittest.TestCase):
    def test_linear_quantile_and_clipping(self):
        raw = torch.arange(1000, dtype=torch.float64).reshape(1, 25, 40)
        image, reference = normalize_sonar_intensity(raw)
        # Linear interpolation at index .998 * 999 = 997.002.
        self.assertAlmostEqual(reference.item(), 997.002)
        self.assertAlmostEqual(image[0, 0, 1].item(), 1.0 / 997.002)
        self.assertEqual(image[0, -1, -1].item(), 1.0)

    def test_isolated_bright_outlier_does_not_dim_the_frame(self):
        raw = torch.ones(1, 25, 40)
        raw[0, -1, -1] = 100.0
        image, reference = normalize_sonar_intensity(raw)
        self.assertEqual(reference.item(), 1.0)
        torch.testing.assert_close(image, torch.ones_like(image))

    def test_training_and_output_use_full_frame_even_for_range_cameras(self):
        raw = torch.tensor([[[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]]])
        training, reference = prepare_sonar_intensity_for_loss(raw)
        package = dict(raw_intensity=raw, untouched="value")
        output = normalize_render_intensity_for_output(
            package, SimpleNamespace(normalizing_method="range")
        )
        torch.testing.assert_close(training, output["r_intensity"])
        torch.testing.assert_close(reference, output["r_intensity_reference"])
        self.assertEqual(reference.shape, (1, 1, 1))
        self.assertLess(training[0, 0, -1].item(), 0.11)
        self.assertNotIn("r_intensity", package)
        self.assertEqual(output["untouched"], "value")

    def test_batch_images_have_independent_references(self):
        raw = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 4)
        batch = torch.stack([raw, raw * 7])
        image, reference = normalize_sonar_intensity(batch)
        self.assertEqual(reference.shape, (2, 1, 1, 1))
        torch.testing.assert_close(image[0], image[1])
        torch.testing.assert_close(reference[1], reference[0] * 7)

    def test_black_frame_and_nonfinite_pixels_have_finite_backward(self):
        raw = torch.zeros(1, 2, 3, requires_grad=True)
        image, reference = normalize_sonar_intensity(raw)
        self.assertEqual(reference.item(), 0.0)
        torch.testing.assert_close(image, torch.zeros_like(image))
        image.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        dirty = torch.tensor([[[float("nan"), float("inf"), -float("inf")],
                               [-1.0, 0.0, 1.0]]])
        image, _ = normalize_sonar_intensity(dirty)
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(((image >= 0) & (image <= 1)).all())

    def test_half_precision_and_quantile_gradient(self):
        raw = torch.linspace(0.1, 1.1, 100, dtype=torch.float16).reshape(1, 10, 10)
        image, _ = normalize_sonar_intensity(raw)
        self.assertEqual(image.dtype, torch.float32)
        values = raw.double().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: prepare_sonar_intensity_for_loss(x)[0], (values,)
        ))


if __name__ == "__main__":
    unittest.main()
