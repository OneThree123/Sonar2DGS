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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static void forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> txImageBuffer,
			std::function<char* (size_t)> imgImageBuffer,
			const int P,
			const float* gs_trans,
			const float* gs_quats,
				const float* gs_scales,
				const float gs_scale_modifier,
				const float lowpass_variance,
				const float* gs_opacities,
			const float* gs_reflections,
			//相机参数
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			//两个方向的图像像素
			const int tx_width, int tx_height,
			const int img_width, int img_height,
			const float fovx, 
			const float fovy,
			const float range_max,
			float* out_intensity,
			float* out_others,
			float* r_raddi,
			int& num_tx_rendered,
			int& num_img_rendered,
			bool debug);

	static void backward(
		const int P,const int num_tx,const int num_img,
		const float* gs_trans,
		const float* gs_quats,
			const float* gs_scales,
			const float gs_scale_modifier,
			const float lowpass_variance,
			const float* gs_opacities,
		const float* gs_reflections,
		const float* viewmatrix,
		const float* projmatrix,
		const float* cam_pos,
		const int tx_width, int tx_height,
		const int img_width, int img_height,
		const float fovx,
		const float fovy,
		const float range_max,
		char* geom_buffer,
		char* binning_buffer,
		char* tx_image_buffer,
		char* img_image_buffer,
		//前向渲染输出
		const float* out_intensity,
		//输入
		const float* dLdintensity,
		const float* dLdothers,
		//中间
			float* dL_dintensities,
				float* dL_dtx_depths,
				float* dL_dimg_params,
			float* dL_dtx_transMat,
			float* dL_dtx_centers,
			float* dL_dtx_axis_u,
			float* dL_dtx_axis_v,
			float* dL_dtx_topk_vis,
		float* dL_dnormals,
		//输出
		float* dLd_gs_trans,
		float* dLd_gs_quats,
		float* dLd_gs_scales,
		float* dLd_gs_opacities,
		float* dLd_gs_reflections,
		float* dLd_gs_screenspace_points,
		bool debug);
	};
};

#endif
