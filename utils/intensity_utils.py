"""Shared sonar intensity processing for training, evaluation and display."""

import torch


INTENSITY_QUANTILE = 0.998
PSNR_PROTOCOL = "mean_of_view_psnr_original_gt_full_frame_q998_clamped_render"


def normalize_sonar_intensity(intensity, eps=1e-8):
    """Map each image's 99.8th percentile to 1 and clip to [0, 1].

    HW/CHW inputs use one reference for the entire image; NCHW inputs use
    one reference per image. The quantile uses linear interpolation and
    remains in the autograd graph. No GT or mask is used to set brightness.
    """
    if intensity.ndim not in (2, 3, 4) or intensity.numel() == 0:
        raise ValueError("Sonar intensity must be a nonempty HW, CHW or NCHW tensor")
    if eps <= 0:
        raise ValueError("Intensity normalization eps must be positive")

    # quantile needs float32/float64; use float32 for half-precision input.
    if intensity.dtype != torch.float64:
        intensity = intensity.float()
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    if intensity.ndim == 4:
        values = intensity.flatten(start_dim=1)
        reference = torch.quantile(values, INTENSITY_QUANTILE, dim=1)
        reference = reference.reshape(-1, 1, 1, 1)
    else:
        reference = torch.quantile(intensity.flatten(), INTENSITY_QUANTILE)
        reference = reference.reshape((1,) * intensity.ndim)

    normalized = intensity / reference.clamp(min=eps)
    return normalized.clamp(0.0, 1.0), reference


def prepare_sonar_intensity_for_loss(intensity):
    """Use the same image normalization as evaluation."""
    return normalize_sonar_intensity(intensity)


def normalize_render_intensity_for_output(render_pkg, viewpoint_camera=None):
    """Attach the shared normalized image, irrespective of camera row modes."""
    image, reference = normalize_sonar_intensity(render_pkg["raw_intensity"])
    output_pkg = dict(render_pkg)
    output_pkg["r_intensity"] = image
    output_pkg["r_intensity_reference"] = reference
    return output_pkg
