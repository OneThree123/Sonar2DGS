#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    reflections,
    opacities,
    scales,
    rotations,
    screenspace_points,
    raster_settings
):
    return _RasterizeGaussians.apply(
        means3D,
        reflections,
        opacities,
        scales,
        rotations,
        screenspace_points,
        raster_settings
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        reflections,
        opacities,
        scales,
        rotations,
        screenspace_points,
        raster_settings,
    ):
        # 与 rasterize_points.h 中 RasterizeGaussiansCUDA 的参数顺序严格一致
        args = (
            means3D,
            reflections,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            raster_settings.lowpass_variance,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.campos,
            raster_settings.tx_height,
            raster_settings.tx_width,
            raster_settings.img_height,
            raster_settings.img_width,
            raster_settings.fovx,
            raster_settings.fovy,
            raster_settings.range_max,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    out_intensity,
                    out_others,
                    r_raddi,
                    geomBuffer,
                    binningBuffer,
                    txImageBuffer,
                    imgImageBuffer,
                    num_tx_rendered,
                    num_img_rendered,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occurred in forward. snapshot_fw.dump was written.")
                raise ex
        else:
            (
                out_intensity,
                out_others,
                r_raddi,
                geomBuffer,
                binningBuffer,
                txImageBuffer,
                imgImageBuffer,
                num_tx_rendered,
                num_img_rendered,
            ) = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_tx_rendered = num_tx_rendered
        ctx.num_img_rendered = num_img_rendered
        ctx.mark_non_differentiable(r_raddi)

        # backward 需要的内容全部存起来
        ctx.save_for_backward(
            means3D,
            reflections,
            opacities,
            scales,
            rotations,
            out_intensity,
            out_others,
            geomBuffer,
            binningBuffer,
            txImageBuffer,
            imgImageBuffer,
        )

        # 对上层只暴露真正需要参与 loss 的两个输出
        return out_intensity, out_others, r_raddi

    @staticmethod
    def backward(ctx, grad_out_intensity, grad_out_others, grad_r_raddi):
        raster_settings = ctx.raster_settings
        num_tx_rendered = ctx.num_tx_rendered
        num_img_rendered = ctx.num_img_rendered

        (
            means3D,
            reflections,
            opacities,
            scales,
            rotations,
            out_intensity,
            out_others,
            geomBuffer,
            binningBuffer,
            txImageBuffer,
            imgImageBuffer,
        ) = ctx.saved_tensors

        # 某个输出没参与 loss 时，PyTorch 可能传 None
        if grad_out_intensity is None:
            grad_out_intensity = torch.zeros_like(out_intensity)
        else:
            grad_out_intensity = grad_out_intensity.contiguous()

        if grad_out_others is None:
            grad_out_others = torch.zeros_like(out_others)
        else:
            grad_out_others = grad_out_others.contiguous()

        # 与 rasterize_points.h 中 RasterizeGaussiansBackwardCUDA 的参数顺序严格一致
        args = (
            means3D,
            reflections,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            raster_settings.lowpass_variance,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.campos,
            raster_settings.tx_height,
            raster_settings.tx_width,
            raster_settings.img_height,
            raster_settings.img_width,
            raster_settings.fovx,
            raster_settings.fovy,
            raster_settings.range_max,
            out_intensity,
            grad_out_intensity,
            grad_out_others,
            geomBuffer,
            num_tx_rendered,
            num_img_rendered,
            binningBuffer,
            txImageBuffer,
            imgImageBuffer,
            raster_settings.debug,
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                (
                    grad_means3D,
                    grad_rotations,
                    grad_scales,
                    grad_opacities,
                    grad_reflections,
                    grad_screenspace_points
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occurred in backward. snapshot_bw.dump was written.")
                raise ex
        else:
            (
                grad_means3D,
                grad_rotations,
                grad_scales,
                grad_opacities,
                grad_reflections,
                grad_screenspace_points
            ) = _C.rasterize_gaussians_backward(*args)

        # 对应 forward 输入顺序：
        # means3D, reflections, opacities, scales, rotations, raster_settings
        return (
            grad_means3D,
            grad_reflections,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_screenspace_points,
            None,
        )


class GaussianRasterizationSettings(NamedTuple):
    tx_height: int
    tx_width: int
    img_height: int
    img_width: int
    fovx: float
    fovy: float
    range_max: float
    scale_modifier: float
    lowpass_variance: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    debug: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(
        self,
        means3D,
        reflections,
        opacities,
        scales,
        rotations,
        screenspace_points
    ):
        return rasterize_gaussians(
            means3D,
            reflections,
            opacities,
            scales,
            rotations,
            screenspace_points,
            self.raster_settings
        )
