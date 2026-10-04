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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <vector>
#include <cmath>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


// __constant__ float2 c_theta_dir[MAX_AZIMUTH_BINS];
// __constant__ float  c_range_val[MAX_RANGE_BINS];

//预计算声呐查找表，声呐像素三角函数和距离查找
void upload_sonar_lookup_tables(
    float theta_min,
    float theta_max,
    int   num_theta,
    float r_min,
    float r_max,
    int   num_range)
{
    std::vector<float2> h_theta_dir(num_theta);
    std::vector<float>  h_range_val(num_range);

    const float dtheta = (theta_max - theta_min) / static_cast<float>(num_theta);
    const float dr     = (r_max - r_min) / static_cast<float>(num_range);

    for (int j = 0; j < num_theta; ++j)
    {
        float theta = theta_min + (j + 0.5f) * dtheta;
        h_theta_dir[j] = make_float2(cosf(theta), sinf(theta));
    }

    for (int i = 0; i < num_range; ++i)
    {
        float r = r_min + (i + 0.5f) * dr;
        h_range_val[i] = r;
    }
    copyConstants2Forward(h_theta_dir.data(), h_range_val.data(), num_theta, num_range);
    copyConstants2Backward(h_theta_dir.data(), h_range_val.data(), num_theta, num_range);
}


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
//生成透射方向排序键
__global__ void duplicateWithKeysTx(
    int P,
    const float4* bounds,
    const float* depths,
    const uint32_t* offsets,
    const uint32_t* tiles_touched,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || tiles_touched[idx] == 0) return;

    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

    uint2 rect_min, rect_max;
    getRectFromBounds(bounds[idx], grid, rect_min, rect_max);

    for (int y = rect_min.y; y < rect_max.y; ++y)
    {
        for (int x = rect_min.x; x < rect_max.x; ++x)
        {
            uint64_t key = (uint64_t)(y * grid.x + x);
            key <<= 32;
            key |= *((const uint32_t*)&depths[idx]); // 正深度，近的小，远的大
            keys_unsorted[off] = key;
            values_unsorted[off] = idx;
            ++off;
        }
    }
}
//生成图像方向分类键
__global__ void duplicateWithKeysImg(
    int P,
    const float4* bounds,
    const uint32_t* offsets,
    const uint32_t* tiles_touched,
    uint64_t* keys_unsorted,
    uint32_t* values_unsorted,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || tiles_touched[idx] == 0) return;

    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

    uint2 rect_min, rect_max;
    getRectFromBounds(bounds[idx], grid, rect_min, rect_max);

    for (int y = rect_min.y; y < rect_max.y; ++y)
    {
        for (int x = rect_min.x; x < rect_max.x; ++x)
        {
            uint64_t key = (uint64_t)(y * grid.x + x);
            key <<= 32;
            key |= (uint64_t)idx;  // 只是保证同 tile 内稳定
            keys_unsorted[off] = key;
            values_unsorted[off] = idx;
            ++off;
        }
    }
}
// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	    GeometryState geom;
	
			    obtain(chunk, geom.tx_bounds, P, 128);
			    obtain(chunk, geom.tx_depths, P, 128);
			    obtain(chunk, geom.tx_centers_2d, P, 128);
			    obtain(chunk, geom.tx_centers, P, 128);
		    obtain(chunk, geom.tx_axis_u, P, 128);
		    obtain(chunk, geom.tx_axis_v, P, 128);
		    obtain(chunk, geom.tx_transMat, P * 9, 128);
		    obtain(chunk, geom.tx_tiles_touched, P, 128);
    obtain(chunk, geom.tx_point_offsets, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.tx_scan_size,
        geom.tx_tiles_touched, geom.tx_point_offsets, P);
    obtain(chunk, geom.tx_scanning_space, geom.tx_scan_size, 128);

    obtain(chunk, geom.img_bounds, P, 128);
    obtain(chunk, geom.img_params, P * SONAR_IMG_PARAM_STRIDE, 128);
    obtain(chunk, geom.img_tiles_touched, P, 128);
    obtain(chunk, geom.img_point_offsets, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.img_scan_size,
        geom.img_tiles_touched, geom.img_point_offsets, P);
    obtain(chunk, geom.img_scanning_space, geom.img_scan_size, 128);

    obtain(chunk, geom.normals, P, 128);
    obtain(chunk, geom.intensities, P, 128);

    return geom;
}

CudaRasterizer::TxImageState CudaRasterizer::TxImageState::fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels)
{
    TxImageState img;
    obtain(chunk, img.ranges, num_tiles, 128);
    obtain(chunk, img.topk_ids, num_pixels * TX_TOPK, 128);
    obtain(chunk, img.topk_vis_weight, num_pixels * TX_TOPK, 128);
    obtain(chunk, img.topk_count, num_pixels, 128);
    return img;
}

CudaRasterizer::ImgImageState CudaRasterizer::ImgImageState::fromChunk(char*& chunk, size_t num_tiles, size_t num_pixels)
{
    ImgImageState img;
    obtain(chunk, img.ranges, num_tiles, 128);
#if IMG_ACCUM_MODE == 0
    obtain(chunk, img.accum_cnt, num_pixels, 128);
#endif
    obtain(chunk, img.hits, num_pixels * TX_TOPK, 128);
    return img;
}


CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
void CudaRasterizer::Rasterizer::forward(
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
    bool debug)
{
    const dim3 tx_grid((tx_width + BLOCK_X - 1) / BLOCK_X,
                       (tx_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 img_grid((img_width + BLOCK_X - 1) / BLOCK_X,
                       (img_height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // 上传声呐查找表
    const float theta_max = fovx/2.0f;
    const float theta_min = -theta_max;
    static bool is_inited = false;

    if (!is_inited) {
        upload_sonar_lookup_tables(theta_min, theta_max, img_width,
                                   0.0f, range_max, img_height);
        is_inited = true;
    }

    // geometry
    size_t geom_chunk_size = required<GeometryState>(P);
    char* geom_chunkptr = geometryBuffer(geom_chunk_size);
    GeometryState geomState = GeometryState::fromChunk(geom_chunkptr, P);

    // image states
    const size_t tx_img_chunk_size  = required<TxImageState>(tx_grid.x*tx_grid.y, tx_width * tx_height);
    char* tx_ptr = txImageBuffer(tx_img_chunk_size);
    TxImageState txImgState = TxImageState::fromChunk(tx_ptr, tx_grid.x*tx_grid.y, tx_width * tx_height);

    const size_t img_img_chunk_size = required<ImgImageState>(img_grid.x*img_grid.y, img_width * img_height);
    char* img_ptr = imgImageBuffer(img_img_chunk_size);
    ImgImageState imgImgState = ImgImageState::fromChunk(img_ptr, img_grid.x*img_grid.y, img_width * img_height);

    // preprocess
    float tan_fovx = tanf(fovx / 2.0f);
    float tan_fovy = tanf(fovy / 2.0f);
    const float inv_dtheta = img_width/fovx;
    const float inv_dr = img_height/range_max;
    CHECK_CUDA(FORWARD::preprocess(
        P,
        (glm::vec3*)gs_trans,
        (glm::vec4*)gs_quats,
	        (glm::vec2*)gs_scales,
	        gs_scale_modifier,
	        lowpass_variance,
	        gs_opacities,
        gs_reflections,
        viewmatrix,
        projmatrix,
        (glm::vec3*)cam_pos,
        tx_width, tx_height,
        tan_fovx, tan_fovy,
        theta_min,
        inv_dtheta,
        0.0f,
        inv_dr,
        range_max,
        tx_grid,
        img_grid,

        geomState.intensities,
        geomState.normals,
        r_raddi,

		        geomState.tx_bounds,
			        geomState.tx_depths,
			        geomState.tx_centers_2d,
			        geomState.tx_centers,
		        geomState.tx_axis_u,
		        geomState.tx_axis_v,
		        geomState.tx_transMat,
		        geomState.tx_tiles_touched,

        geomState.img_bounds,
        geomState.img_params,
        geomState.img_tiles_touched
    ), debug);

    // tx prefix sum
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.tx_scanning_space,
        geomState.tx_scan_size,
        geomState.tx_tiles_touched,
        geomState.tx_point_offsets,
        P), debug);

    // img prefix sum
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.img_scanning_space,
        geomState.img_scan_size,
        geomState.img_tiles_touched,
        geomState.img_point_offsets,
        P), debug);

    CHECK_CUDA(cudaMemcpy(&num_tx_rendered,
        geomState.tx_point_offsets + P - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug);
    CHECK_CUDA(cudaMemcpy(&num_img_rendered,
        geomState.img_point_offsets + P - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug);

    // 两套 binning buffer
    size_t tx_binning_size = required<BinningState>(max(1, num_tx_rendered));
    size_t img_binning_size = required<BinningState>(max(1, num_img_rendered));
    char* bin_chunkptr = binningBuffer(tx_binning_size + img_binning_size);

    char* tx_bin_ptr = bin_chunkptr;
    BinningState txBinning = BinningState::fromChunk(tx_bin_ptr, max(1, num_tx_rendered));
    BinningState imgBinning = BinningState::fromChunk(tx_bin_ptr, max(1, num_img_rendered));

    const int tx_bit  = getHigherMsb(tx_grid.x * tx_grid.y);
    const int img_bit = getHigherMsb(img_grid.x * img_grid.y);

	CHECK_CUDA(cudaMemset(txImgState.topk_count, 0,
		tx_width * tx_height * sizeof(uint8_t)), debug);
    CHECK_CUDA(cudaMemset(out_others, 0, OTHERS_NUM * tx_width * tx_height * sizeof(float)), debug);


    // tx duplicate + sort + ranges
    if (num_tx_rendered > 0)
    {
        duplicateWithKeysTx<<<(P + 255) / 256, 256>>>(
            P,
            geomState.tx_bounds,
            geomState.tx_depths,
            geomState.tx_point_offsets,
            geomState.tx_tiles_touched,
            txBinning.point_list_keys_unsorted,
            txBinning.point_list_unsorted,
            tx_grid);

        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            txBinning.list_sorting_space,
            txBinning.sorting_size,
            txBinning.point_list_keys_unsorted, txBinning.point_list_keys,
            txBinning.point_list_unsorted, txBinning.point_list,
            num_tx_rendered, 0, 32 + tx_bit), debug);

        CHECK_CUDA(cudaMemset(txImgState.ranges, 0,
            tx_grid.x * tx_grid.y * sizeof(uint2)), debug);

        identifyTileRanges<<<(num_tx_rendered + 255) / 256, 256>>>(
            num_tx_rendered,
            txBinning.point_list_keys,
            txImgState.ranges);


        CHECK_CUDA(FORWARD::render_tx(
            tx_grid, block,
            txImgState.ranges,
            txBinning.point_list,
	            tx_width, tx_height,
		            range_max,
				            (glm::vec3*)cam_pos,
				            geomState.tx_transMat,
				            geomState.tx_centers_2d,
				            geomState.tx_centers,
		            geomState.tx_axis_u,
		            geomState.tx_axis_v,
			            geomState.normals,
		            gs_opacities,
		            lowpass_variance,
	            txImgState.topk_ids,
            txImgState.topk_vis_weight,
            txImgState.topk_count,
            out_others
        ), debug);
    }
	
	CHECK_CUDA(cudaMemset(out_intensity, 0, img_width * img_height * sizeof(float)), debug);
    // img duplicate + sort + ranges
    if (num_img_rendered > 0)
    {
        duplicateWithKeysImg<<<(P + 255) / 256, 256>>>(
            P,
            geomState.img_bounds,
            geomState.img_point_offsets,
            geomState.img_tiles_touched,
            imgBinning.point_list_keys_unsorted,
            imgBinning.point_list_unsorted,
            img_grid);

        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
            imgBinning.list_sorting_space,
            imgBinning.sorting_size,
            imgBinning.point_list_keys_unsorted, imgBinning.point_list_keys,
            imgBinning.point_list_unsorted, imgBinning.point_list,
            num_img_rendered, 0, 32 + img_bit), debug);

        CHECK_CUDA(cudaMemset(imgImgState.ranges, 0,
            img_grid.x * img_grid.y * sizeof(uint2)), debug);

        identifyTileRanges<<<(num_img_rendered + 255) / 256, 256>>>(
            num_img_rendered,
            imgBinning.point_list_keys,
            imgImgState.ranges);

        CHECK_CUDA(FORWARD::render_img(
            img_grid, block,
            imgImgState.ranges,
            imgBinning.point_list,
            tx_width, tx_height,
            img_width, img_height,
            geomState.img_params,
            geomState.tx_transMat,
            geomState.intensities,
            geomState.normals,
            txImgState.topk_ids,
            txImgState.topk_vis_weight,
            txImgState.topk_count,
            out_intensity,
        #if IMG_ACCUM_MODE == 0
            imgImgState.accum_cnt
        #else
            nullptr
        #endif
        ), debug);
    }
}

// Produce necessary gradients for optimization, corresponding
// to sonar forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	bool debug)
{
    const dim3 tx_grid((tx_width + BLOCK_X - 1) / BLOCK_X, (tx_height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 img_grid((img_width + BLOCK_X - 1) / BLOCK_X, (img_height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	TxImageState txImgState = TxImageState::fromChunk(tx_image_buffer, tx_grid.x * tx_grid.y , tx_width * tx_height);
	ImgImageState imgImgState = ImgImageState::fromChunk(img_image_buffer, img_grid.x * img_grid.y, img_width * img_height);
    
	char* tx_bin_ptr = binning_buffer;
	BinningState txBinning = BinningState::fromChunk(tx_bin_ptr, max(1, num_tx));
	BinningState imgBinning = BinningState::fromChunk(tx_bin_ptr, max(1, num_img));

    if (num_img > 0)
    {
        //图像反传
        CHECK_CUDA(BACKWARD::render_img(
            img_grid, block,
            imgImgState.ranges,
            imgBinning.point_list,
            tx_width, tx_height,
            img_width, img_height,
            geomState.img_params,
            geomState.tx_transMat,
            geomState.intensities,
            txImgState.topk_ids,
            txImgState.topk_vis_weight,
            txImgState.topk_count,
            dLdintensity,
        #if IMG_ACCUM_MODE == 0
            imgImgState.accum_cnt,
            nullptr,
        #elif IMG_ACCUM_MODE == 1
            nullptr,
            nullptr,
        #elif IMG_ACCUM_MODE == 2
            nullptr,
            out_intensity,
        #endif
            dL_dimg_params,
            dL_dtx_transMat,
            dL_dintensities,
            dL_dtx_topk_vis), debug);
    }

    if (num_tx > 0)
    {
        //透射反传
        CHECK_CUDA(BACKWARD::render_tx(
            tx_grid, block,
            tx_width, tx_height,
		            range_max,
			            (const glm::vec3*)cam_pos,
			            geomState.tx_transMat,
			            geomState.tx_centers_2d,
			            geomState.tx_depths,
		            geomState.tx_centers,
		            geomState.tx_axis_u,
		            geomState.tx_axis_v,
		            geomState.normals,
	            gs_opacities,
	            lowpass_variance,
	            txImgState.topk_ids,
            txImgState.topk_vis_weight,
            txImgState.topk_count,
            //输入
            dLdothers,
            dL_dtx_topk_vis,
            //输出
		            dL_dtx_transMat,
		            dL_dtx_depths,
		            dL_dtx_centers,
		            dL_dtx_axis_u,
		            dL_dtx_axis_v,
		            dL_dnormals,
            dLd_gs_opacities), debug);
    }

    //参数反传
	CHECK_CUDA(BACKWARD::preprocess(
		P,
		(const glm::vec3*)gs_trans,
		(const glm::vec4*)gs_quats,
		(const glm::vec2*)gs_scales,
		gs_scale_modifier,
		gs_reflections,
		viewmatrix,
		projmatrix,
		(const glm::vec3*)cam_pos,
		tx_width, tx_height,
		img_width, img_height,
		fovx,
		range_max,
		dL_dnormals,
		dL_dintensities,
			dL_dtx_transMat,
			dL_dtx_depths,
			dL_dtx_centers,
			dL_dtx_axis_u,
			dL_dtx_axis_v,
			dL_dimg_params,
		(glm::vec3*)dLd_gs_trans,
		(glm::vec4*)dLd_gs_quats,
		(glm::vec2*)dLd_gs_scales,
		dLd_gs_reflections,
		(glm::vec2*)dLd_gs_screenspace_points), debug);
}
