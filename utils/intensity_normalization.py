"""Scale test renders using the latest complete training batch's multiplier."""

import math

import torch


TEST_RENDER_NORMALIZATION_CONFIG = {
    "psnr_protocol": "mean_of_view_psnr_original_gt_full_image_latest_batch_mean_clamped_render_v1",
    "render_normalization": "latest_batch_color_scale_mean",
    "render_scale_reduction": "arithmetic_mean_of_all_applied_training_coefficients",
    "render_scale_source": "latest_complete_training_group",
}


def normalize_test_intensity(intensity, scale_mean):
    """Return clamp(clean intensity * scale_mean, 0, 1) and the multiplier.

    The multiplier is shared by all test views, independent of their pixels,
    ground truths and camera normalization methods. Non-finite/negative pixels
    become zero. Input is never modified; half precision promotes to float32.
    """
    if intensity.ndim not in (2, 3) or (intensity.ndim == 3 and intensity.shape[0] != 1):
        raise ValueError(f"Expected a grayscale [H, W] or [1, H, W] frame, got {tuple(intensity.shape)}")
    if intensity.numel() == 0:
        raise ValueError("Cannot normalize an empty test frame")
    if not intensity.is_floating_point():
        raise TypeError("Test intensity must be a floating-point tensor")
    if scale_mean is None or not math.isfinite(scale_mean) or scale_mean < 0:
        raise ValueError("Test rendering requires a finite, non-negative latest batch color scale mean")

    if intensity.dtype in (torch.float16, torch.bfloat16):
        intensity = intensity.float()
    clean_intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    image = (clean_intensity * scale_mean).clamp(0.0, 1.0)
    return image, scale_mean
