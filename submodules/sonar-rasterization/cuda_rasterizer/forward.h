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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

void copyConstants2Forward(float2* d_theta_dir, float* d_range_val, float num_theta, float num_range);


namespace FORWARD
{
void preprocess(
	int P,
	/*存储高斯列表*/
	const glm::vec3* gs_trans,
	const glm::vec4* gs_quats,//四元数
		const glm::vec2* gs_scales,
		const float gs_scale_modifier,
		const float lowpass_variance,
	    const float* gs_opacities,
	const float* gs_reflections,
	/*相机参数*/
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int tx_width, int tx_height,
	//视野角度相关参数，需要在外部完成计算
	const float tan_fovx, const float tan_fovy,
	const float theta_min,
	const float inv_dtheta,
	const float range_min,
	const float inv_dr,
	const float range_max,
	//成像网格，需要在外部完成计算
	const dim3 tx_grid,
	const dim3 img_grid,
	/*输出*/
	float* gs_intensities,
	float3* gs_normals,
    float* r_raddi,
		/*输出：tx*/
				float4* tx_bounds,
				float* tx_depths,
				float2* tx_centers_2d,
				float3* tx_centers,
				float3* tx_axis_u,
			float3* tx_axis_v,
			float* tx_transMats,//其实是gs2tx的变化矩阵，但是后续用来算tx2gs
			uint32_t* tx_tiles_touched,
	/*输出：img*/
	float4* img_bounds,
	float* img_params,//压根不是变换矩阵，用来辅助img到gs的计算
	uint32_t* img_tiles_touched
);

void render_tx(
    const dim3 grid, dim3 block,
    const uint2*  ranges,
    const uint32_t*  point_list,
	    int W, int H,
	    const float range_max,
	    const glm::vec3* cam_pos,
	    const float*  tx_transMats,
	    const float2* tx_centers_2d,
	    const float3* tx_centers,
		    const float3* tx_axis_u,
		    const float3* tx_axis_v,
		    const float3* gs_normals,
	    const float*  opacity,
	    const float lowpass_variance,
	    uint32_t*  topk_ids,
    float*  topk_vis_weight,
    uint8_t*  topk_count,
	float* out_others);

void render_img(
	const dim3 grid, dim3 block,
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int tx_width, int tx_height,
    const int img_width, int img_height,
    const float* __restrict__ img_params,     // P * 11
    const float* __restrict__ tx_transMats,   // P * 9
    const float* __restrict__ intensity,      // P
    const float3* __restrict__ normal,        // P，可先不用，但保留独立接口
    const uint32_t* __restrict__ tx_topk_ids,
    const float* __restrict__ tx_topk_vis_weight,
    const uint8_t* __restrict__ tx_topk_count,
    float* __restrict__ out_intensity,
	int* __restrict__ diff_assist
	);
}


#endif
