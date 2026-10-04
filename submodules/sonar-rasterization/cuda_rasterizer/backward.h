/*
 * Sonar-specific backward interface for modified 2DGS.
 */
#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "config.h"

void copyConstants2Backward(float2* d_theta_dir, float* d_range_val, float num_theta, float num_range);


namespace BACKWARD
{
    void render_img(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int tx_width, int tx_height,
        int img_width, int img_height,
        const float* img_params,
        const float* tx_transMats,
        const float* intensities,
        const uint32_t* tx_topk_ids,
        const float* tx_topk_vis_weight,
        const uint8_t* tx_topk_count,
        const float* dL_dout_intensity,
        const int* accum_cnt,
        const float* out_intensity,
        float* dL_dimg_params,
        float* dL_dtx_transMat,
        float* dL_dintensity,
        float* dL_dtx_topk_vis_weight);

    void render_tx(
        const dim3 grid, dim3 block,
	        int W, int H,
		        float range_max,
			        const glm::vec3* cam_pos,
			        const float* tx_transMats,
			        const float2* tx_centers_2d,
			        const float* tx_depths,
		        const float3* tx_centers,
		        const float3* tx_axis_u,
		        const float3* tx_axis_v,
		        const float3* gs_normals,
	        const float* opacity,
	        const float lowpass_variance,
	        const uint32_t* topk_ids,
        const float* topk_vis_weight,
        const uint8_t* topk_count,
        //输入
        const float* dL_dothers,
        const float* dL_dtx_topk_vis_weight,
	        //输出
		        float* dL_dtx_transMat,
		        float* dL_dtx_depth,
		        float* dL_dtx_center,
		        float* dL_dtx_axis_u,
		        float* dL_dtx_axis_v,
		        float* dL_dnormal3D,
        float* dL_dopacity);

    void preprocess(
        int P,
        const glm::vec3* gs_trans,
        const glm::vec4* gs_quats,
        const glm::vec2* gs_scales,
        float gs_scale_modifier,
        const float* gs_reflections,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        int tx_width, int tx_height,
        int img_width, int img_height,
        float fovx,
        float range_max,
        const float* dL_dnormal3D,
        const float* dL_dintensity,
		        const float* dL_dtx_transMat,
		        const float* dL_dtx_depth,
		        const float* dL_dtx_center,
		        const float* dL_dtx_axis_u,
		        const float* dL_dtx_axis_v,
		        const float* dL_dimg_params,
        glm::vec3* dL_dgs_trans,
        glm::vec4* dL_dgs_quats,
        glm::vec2* dL_dgs_scales,
        float* dL_dgs_reflections, 
        glm::vec2* dL_dgs_screenspace_points);
}

#endif
