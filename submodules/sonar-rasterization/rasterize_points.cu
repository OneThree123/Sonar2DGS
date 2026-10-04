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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}
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
    const bool debug)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3)
        TORCH_CHECK(false, "means3D must have shape [P, 3]");

    const int P = means3D.size(0);

    CHECK_INPUT(means3D);
    CHECK_INPUT(reflections);
    CHECK_INPUT(opacity);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);
    TORCH_CHECK(lowpass_variance > 0.0f, "lowpass_variance must be positive");

    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto byte_opts  = torch::TensorOptions().device(means3D.device()).dtype(torch::kUInt8);

    torch::Tensor out_intensity = torch::zeros({1, img_height, img_width}, float_opts);
    torch::Tensor out_others    = torch::zeros({OTHERS_NUM, tx_height, tx_width}, float_opts);
    torch::Tensor r_raddi   = torch::zeros({P, 1}, float_opts);

    torch::Tensor geomBuffer    = torch::empty({0}, byte_opts);
    torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
    torch::Tensor txImageBuffer = torch::empty({0}, byte_opts);
    torch::Tensor imgImageBuffer= torch::empty({0}, byte_opts);

    auto geomFunc = resizeFunctional(geomBuffer);
    auto binFunc  = resizeFunctional(binningBuffer);
    auto txFunc   = resizeFunctional(txImageBuffer);
    auto imgFunc  = resizeFunctional(imgImageBuffer);

    int num_tx_rendered = 0;
    int num_img_rendered = 0;

    if (P > 0) {
        CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binFunc,
            txFunc,
            imgFunc,
            P,
            means3D.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            lowpass_variance,
            opacity.contiguous().data_ptr<float>(),
            reflections.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tx_width, tx_height,
            img_width, img_height,
            fovx, fovy, range_max,
            out_intensity.contiguous().data_ptr<float>(),
            out_others.contiguous().data_ptr<float>(),
            r_raddi.contiguous().data_ptr<float>(),
            num_tx_rendered,
            num_img_rendered,
            debug);
    }

    return std::make_tuple(
        out_intensity, out_others, r_raddi, 
        geomBuffer, binningBuffer,
        txImageBuffer, imgImageBuffer,
        num_tx_rendered, num_img_rendered);
}

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
    const bool debug)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = means3D.size(0);

    CHECK_INPUT(means3D);
    CHECK_INPUT(reflections);
    CHECK_INPUT(opacity);
    CHECK_INPUT(scales);
    CHECK_INPUT(rotations);
    CHECK_INPUT(viewmatrix);
    CHECK_INPUT(projmatrix);
    CHECK_INPUT(campos);
    CHECK_INPUT(out_intensity);
    CHECK_INPUT(dLdintensity);
    CHECK_INPUT(dLdothers);
    CHECK_INPUT(geomBuffer);
    CHECK_INPUT(binningBuffer);
    CHECK_INPUT(txImageBuffer);
    CHECK_INPUT(imgImageBuffer);
    TORCH_CHECK(lowpass_variance > 0.0f, "lowpass_variance must be positive");

    // 可选但很建议：做几条最关键的 shape 检查
    TORCH_CHECK(scales.ndimension() == 2 && scales.size(0) == P && scales.size(1) == 2,
                "scales must have shape [P, 2]");
    TORCH_CHECK(rotations.ndimension() == 2 && rotations.size(0) == P && rotations.size(1) == 4,
                "rotations must have shape [P, 4]");
    TORCH_CHECK(out_intensity.ndimension() == 3 &&
                out_intensity.size(0) == 1 &&
                out_intensity.size(1) == img_height &&
                out_intensity.size(2) == img_width,
                "out_intensity must have shape [1, img_height, img_width]");
    TORCH_CHECK(dLdintensity.ndimension() == 3 &&
                dLdintensity.size(0) == 1 &&
                dLdintensity.size(1) == img_height &&
                dLdintensity.size(2) == img_width,
                "dLdintensity must have shape [1, img_height, img_width]");
    TORCH_CHECK(dLdothers.ndimension() == 3 &&
                dLdothers.size(0) == OTHERS_NUM &&
                dLdothers.size(1) == tx_height &&
                dLdothers.size(2) == tx_width,
                "dLdothers must have shape [OTHERS_NUM, tx_height, tx_width]");

    auto float_opts = means3D.options().dtype(torch::kFloat32);

    // 最终输出梯度
    torch::Tensor dLd_gs_trans       = torch::zeros({P, 3}, float_opts);
    torch::Tensor dLd_gs_quats       = torch::zeros({P, 4}, float_opts);
    torch::Tensor dLd_gs_scales      = torch::zeros({P, 2}, float_opts);
    torch::Tensor dLd_gs_opacities   = torch::zeros_like(opacity);
    torch::Tensor dLd_gs_reflections = torch::zeros_like(reflections);
    torch::Tensor dLd_gs_screenspace_points = torch::zeros({P, 2}, float_opts);

    // backward 中间梯度缓冲
	    torch::Tensor dL_dintensities = torch::zeros({P}, float_opts);
	    torch::Tensor dL_dtx_depths   = torch::zeros({P}, float_opts);
	    torch::Tensor dL_dimg_params  = torch::zeros({P, SONAR_IMG_PARAM_STRIDE}, float_opts);
    torch::Tensor dL_dtx_transMat = torch::zeros({P, 9}, float_opts);
    torch::Tensor dL_dtx_centers  = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dtx_axis_u   = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dtx_axis_v   = torch::zeros({P, 3}, float_opts);
    torch::Tensor dL_dtx_topk_vis = torch::zeros({tx_height * tx_width, TX_TOPK}, float_opts);
    torch::Tensor dL_dnormals     = torch::zeros({P, 3}, float_opts);

    if (P == 0) {
        return std::make_tuple(
            dLd_gs_trans,
            dLd_gs_quats,
            dLd_gs_scales,
            dLd_gs_opacities,
            dLd_gs_reflections,
            dLd_gs_screenspace_points
        );
    }

    // 持有 contiguous 副本，避免临时对象生命周期问题
    auto means3D_c       = means3D.contiguous();
    auto reflections_c   = reflections.contiguous();
    auto opacity_c       = opacity.contiguous();
    auto scales_c        = scales.contiguous();
    auto rotations_c     = rotations.contiguous();
    auto viewmatrix_c    = viewmatrix.contiguous();
    auto projmatrix_c    = projmatrix.contiguous();
    auto campos_c        = campos.contiguous();
    auto out_intensity_c = out_intensity.contiguous();
    auto dLdintensity_c  = dLdintensity.contiguous();
    auto dLdothers_c     = dLdothers.contiguous();
    auto geomBuffer_c    = geomBuffer.contiguous();
    auto binningBuffer_c = binningBuffer.contiguous();
    auto txImageBuffer_c = txImageBuffer.contiguous();
    auto imgImageBuffer_c= imgImageBuffer.contiguous();

    CudaRasterizer::Rasterizer::backward(
        P,
        num_tx_rendered,
        num_img_rendered,
        means3D_c.data_ptr<float>(),
        rotations_c.data_ptr<float>(),
        scales_c.data_ptr<float>(),
        scale_modifier,
        lowpass_variance,
        opacity_c.data_ptr<float>(),
        reflections_c.data_ptr<float>(),
        viewmatrix_c.data_ptr<float>(),
        projmatrix_c.data_ptr<float>(),
        campos_c.data_ptr<float>(),
        tx_width,
        tx_height,
        img_width,
        img_height,
        fovx,
        fovy,
        range_max,
        reinterpret_cast<char*>(geomBuffer_c.data_ptr<uint8_t>()),
        reinterpret_cast<char*>(binningBuffer_c.data_ptr<uint8_t>()),
        reinterpret_cast<char*>(txImageBuffer_c.data_ptr<uint8_t>()),
        reinterpret_cast<char*>(imgImageBuffer_c.data_ptr<uint8_t>()),
        out_intensity_c.data_ptr<float>(),
        dLdintensity_c.data_ptr<float>(),
	        dLdothers_c.data_ptr<float>(),
	        dL_dintensities.data_ptr<float>(),
	        dL_dtx_depths.data_ptr<float>(),
	        dL_dimg_params.data_ptr<float>(),
        dL_dtx_transMat.data_ptr<float>(),
        dL_dtx_centers.data_ptr<float>(),
        dL_dtx_axis_u.data_ptr<float>(),
        dL_dtx_axis_v.data_ptr<float>(),
        dL_dtx_topk_vis.data_ptr<float>(),
        dL_dnormals.data_ptr<float>(),
        dLd_gs_trans.data_ptr<float>(),
        dLd_gs_quats.data_ptr<float>(),
        dLd_gs_scales.data_ptr<float>(),
        dLd_gs_opacities.data_ptr<float>(),
        dLd_gs_reflections.data_ptr<float>(),
        dLd_gs_screenspace_points.data_ptr<float>(),
        debug
    );

    return std::make_tuple(
        dLd_gs_trans,
        dLd_gs_quats,
        dLd_gs_scales,
        dLd_gs_opacities,
        dLd_gs_reflections,
        dLd_gs_screenspace_points
    );
}
