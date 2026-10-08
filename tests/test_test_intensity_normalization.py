"""Numerical checks for test frames scaled by the latest training batch mean."""

import unittest

import torch

from utils.intensity_normalization import normalize_test_intensity


class TestIntensityNormalizationTest(unittest.TestCase):
    def test_applies_training_multiplier_and_clips_without_modifying_input(self):
        intensity = torch.tensor([[0., 1.], [2., 100.]], dtype=torch.float64)
        original = intensity.clone()
        image, scale = normalize_test_intensity(intensity, .25)
        self.assertEqual(scale, .25)
        torch.testing.assert_close(image, torch.tensor([[0., .25], [.5, 1.]], dtype=torch.float64))
        torch.testing.assert_close(intensity, original)
        self.assertEqual(image.dtype, intensity.dtype)
        self.assertEqual(image.shape, intensity.shape)

    def test_sparse_pixels_and_range_rows_use_one_fixed_multiplier(self):
        intensity = torch.tensor([[[0., 2.], [10., 20.]]])
        image, _ = normalize_test_intensity(intensity, .01)
        torch.testing.assert_close(image, intensity * .01)
        brighter, _ = normalize_test_intensity(intensity * 2, .01)
        torch.testing.assert_close(brighter, image * 2)

    def test_zero_multiplier_and_black_frames(self):
        for raw, scale in ((0., 1e8), (7., 0.)):
            with self.subTest(raw=raw, scale=scale):
                image, _ = normalize_test_intensity(torch.full((1, 3, 3), raw), scale)
                torch.testing.assert_close(image, torch.zeros_like(image))

    def test_nonfinite_and_negative_pixels_become_zero(self):
        intensity = torch.tensor([[float("nan"), float("inf"), float("-inf"), -2., 1.]])
        image, _ = normalize_test_intensity(intensity, .25)
        torch.testing.assert_close(image, torch.tensor([[0., 0., 0., 0., .25]]))
        self.assertTrue(torch.isnan(intensity[0, 0]))
        self.assertTrue(torch.isinf(intensity[0, 1]))
        self.assertEqual(intensity[0, 3].item(), -2.)

    def test_half_precision_promotes_to_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                intensity = torch.tensor([[0., 1., 2., 4.]], dtype=dtype)
                image, scale = normalize_test_intensity(intensity, 1e8)
                expected, _ = normalize_test_intensity(intensity.float(), 1e8)
                self.assertEqual(image.dtype, torch.float32)
                self.assertEqual(scale, 1e8)
                torch.testing.assert_close(image, expected)
                self.assertTrue(torch.isfinite(image).all())

    def test_noncontiguous_frame(self):
        intensity = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected, _ = normalize_test_intensity(intensity, .05)
        image, _ = normalize_test_intensity(intensity.T, .05)
        torch.testing.assert_close(image, expected.T)

    def test_invalid_frames_and_missing_or_invalid_means(self):
        for shape in ((4,), (3, 2, 2), (2, 1, 2, 2), (1, 0, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                normalize_test_intensity(torch.zeros(shape), 1.)
        for dtype in (torch.int32, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                normalize_test_intensity(torch.zeros(2, 2, dtype=dtype), 1.)
        for scale in (None, -.1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                normalize_test_intensity(torch.ones(2, 2), scale)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_agree_and_device_is_preserved(self):
        intensity = torch.arange(400, dtype=torch.float32).reshape(1, 20, 20)
        cpu_image, _ = normalize_test_intensity(intensity, .002)
        cuda_image, _ = normalize_test_intensity(intensity.cuda(), .002)
        self.assertEqual(cuda_image.device.type, "cuda")
        torch.testing.assert_close(cuda_image.cpu(), cpu_image)


if __name__ == "__main__":
    unittest.main()
