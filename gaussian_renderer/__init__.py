import torch

from sonar_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal_masked


def normalize_sonar_intensity(intensity, method="all", eps=1e-8):
    """
    Sonar intensity normalization for output/display.

    The default "all" method uses one max value for the full frame. The
    "range" method uses one max value for each height/range bin.
    """
    method = str(method or "all").lower()
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "all":
        max_intensity = torch.amax(intensity, dim=(-2, -1), keepdim=True)
    elif method == "range":
        max_intensity = torch.amax(intensity, dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown sonar intensity normalizing method: {method!r}")

    normalized_intensity = intensity / torch.clamp(max_intensity, min=eps)
    normalized_intensity = torch.nan_to_num(
        normalized_intensity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return normalized_intensity, max_intensity


def match_sonar_intensity_mean(intensity, gt_image, valid_mask, method="all", eps=1e-8):
    """
    Scale rendered intensity for image loss while measuring means in valid areas.

    The scale is applied to all pixels. The "all" method computes one scale for
    the full image; the "range" method computes one scale per height/range bin.
    """
    method = str(method or "all").lower()
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    gt_image = torch.nan_to_num(gt_image, nan=0.0, posinf=0.0, neginf=0.0)

    if valid_mask is None:
        valid_mask = torch.ones_like(gt_image)
    else:
        valid_mask = torch.clamp(valid_mask, 0.0, 1.0)

    gt_image = gt_image.to(device=intensity.device, dtype=intensity.dtype)
    valid_mask = valid_mask.to(device=intensity.device, dtype=intensity.dtype)
    while gt_image.ndim < intensity.ndim:
        gt_image = gt_image.unsqueeze(0)
    while valid_mask.ndim < intensity.ndim:
        valid_mask = valid_mask.unsqueeze(0)
    if valid_mask.shape != intensity.shape:
        valid_mask = valid_mask.expand_as(intensity)

    if method == "all":
        reduce_dim = (-2, -1)
    elif method == "range":
        reduce_dim = -1
    else:
        raise ValueError(f"Unknown sonar intensity normalizing method: {method!r}")

    valid_weight = valid_mask.sum(dim=reduce_dim, keepdim=True)
    safe_weight = torch.clamp(valid_weight, min=eps)
    render_mean = (intensity * valid_mask).sum(dim=reduce_dim, keepdim=True) / safe_weight
    gt_mean = (gt_image * valid_mask).sum(dim=reduce_dim, keepdim=True) / safe_weight

    scale = gt_mean / torch.clamp(render_mean, min=eps)
    scale = torch.where(valid_weight > eps, scale, torch.ones_like(scale))
    matched_intensity = intensity * scale
    matched_intensity = torch.nan_to_num(
        matched_intensity,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return matched_intensity, scale


def prepare_sonar_intensity_for_loss(
    intensity,
    gt_image,
    valid_mask,
    method="all",
    loss_mode="normalized",
):
    loss_mode = str(loss_mode or "normalized").lower()
    if loss_mode == "normalized":
        return normalize_sonar_intensity(intensity, method=method)
    if loss_mode == "mean_match":
        return match_sonar_intensity_mean(intensity, gt_image, valid_mask, method=method)
    raise ValueError(
        "Unknown loss intensity mode: "
        f"{loss_mode!r}. Expected 'normalized' or 'mean_match'."
    )


def normalize_render_intensity_for_output(render_pkg, viewpoint_camera):
    """Return render package with display-normalized sonar intensity attached."""
    r_intensity, r_intensity_max = normalize_sonar_intensity(
        render_pkg["raw_intensity"],
        method=getattr(viewpoint_camera, "normalizing_method", "all"),
    )
    output_pkg = dict(render_pkg)
    output_pkg["r_intensity"] = r_intensity
    output_pkg["r_intensity_max"] = r_intensity_max
    return output_pkg


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    scaling_modifier=1.0,
    lowpass_variance=0.5,
):
    raster_settings = GaussianRasterizationSettings(
        tx_height=int(viewpoint_camera.tx_height),
        tx_width=int(viewpoint_camera.tx_width),
        img_height=int(viewpoint_camera.img_height),
        img_width=int(viewpoint_camera.img_width),
        fovx=float(viewpoint_camera.FoVx),
        fovy=float(viewpoint_camera.FoVy),
        range_max=float(viewpoint_camera.range_max),
        scale_modifier=scaling_modifier,
        lowpass_variance=float(lowpass_variance),
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        debug=getattr(pipe, "debug", False),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    #定义存储对图像平面投影梯度的容器，损失对图像平面梯度用于增删点云控制
    screenspace_points = torch.zeros(
        (pc.get_xyz.shape[0], 2),
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device=pc.get_xyz.device,
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raw_intensity, r_others, r_raddi = rasterizer(
        means3D=pc.get_xyz,
        reflections=pc.get_reflection,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        screenspace_points = screenspace_points,
    )

    if r_raddi.ndim == 2 and r_raddi.shape[-1] == 1:
        r_raddi = r_raddi.squeeze(-1)


    r_depth = r_others[0:1]
    r_alpha = r_others[1:2]
    r_normal = r_others[2:5]
    r_dist = r_others[5:6]

    r_depth = r_depth / torch.clamp(r_alpha.detach(), min=1e-8)
    r_depth = torch.nan_to_num(r_depth, nan=0.0, posinf=0.0, neginf=0.0)

    surf_normal, surf_normal_mask = depth_to_normal_masked(
        viewpoint_camera,
        r_depth,
        r_alpha.detach() > 0.8,
    )
    surf_normal = surf_normal.permute(2, 0, 1)
    surf_normal_mask = surf_normal_mask[None]

    return {
        "raw_intensity": raw_intensity,
        "screenspace_points": screenspace_points,
        "gs_visibility": r_raddi > 0,
        "gs_raddi": r_raddi,
        "rend_alpha": r_alpha,
        "rend_normal": r_normal,
        "rend_dist": r_dist,
        "rend_depth": r_depth,
        "surf_normal": surf_normal,
        "surf_normal_mask": surf_normal_mask,
    }
