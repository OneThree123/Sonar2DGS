import torch

from sonar_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.point_utils import depth_to_normal_masked
from utils.intensity_utils import (
    normalize_sonar_intensity,
    normalize_render_intensity_for_output,
    prepare_sonar_intensity_for_loss,
)


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
