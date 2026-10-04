/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, 
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor,
    int, int>
RasterizeGaussiansCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& reflections,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const float lowpass_variance,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const int tx_height,
    const int tx_width,
    const int img_height,
    const int img_width,
    const float fovx,
    const float fovy,
    const float range_max,
    const bool debug);

std::tuple<
    torch::Tensor, // dLd_gs_trans
    torch::Tensor, // dLd_gs_quats
    torch::Tensor, // dLd_gs_scales
    torch::Tensor, // dLd_gs_opacities
    torch::Tensor, // dLd_gs_reflections
    torch::Tensor  // dLd_gs_screenspace_points
>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& reflections,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const float lowpass_variance,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& campos,
    const int tx_height,
    const int tx_width,
    const int img_height,
    const int img_width,
    const float fovx,
    const float fovy,
    const float range_max,
    const torch::Tensor& out_intensity,
    const torch::Tensor& dLdintensity,
    const torch::Tensor& dLdothers,
    const torch::Tensor& geomBuffer,
    const int num_tx_rendered,
    const int num_img_rendered,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& txImageBuffer,
    const torch::Tensor& imgImageBuffer,
    const bool debug);
		
