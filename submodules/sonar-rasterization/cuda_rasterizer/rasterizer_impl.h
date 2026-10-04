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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
			// tx
			size_t tx_scan_size;
					float4* tx_bounds; 
					float* tx_depths;
					float2* tx_centers_2d;
					float3* tx_centers;
				float3* tx_axis_u;
				float3* tx_axis_v;
				float* tx_transMat;            // P * 9
				uint32_t* tx_tiles_touched;
		uint32_t* tx_point_offsets;
		char* tx_scanning_space;

		// img
		size_t img_scan_size;
		float4* img_bounds;
		float* img_params;           // P *11
		uint32_t* img_tiles_touched;
		uint32_t* img_point_offsets;
		char* img_scanning_space;

		// neutral
		float3* normals;
		float* intensities;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct TxImageState
	{
		uint2* ranges;

		uint32_t* topk_ids;      // N * TX_TOPK
		float* topk_vis_weight;         // N * TX_TOPK
		uint8_t* topk_count;     // N

		static TxImageState fromChunk(char*& chunk, size_t N1, size_t N2);
	};
	struct ImgHit
	{
		uint32_t gid;
		int32_t idx00, idx10, idx01, idx11;
		float u, v;
		float tx_x, tx_y;
		float vis;
		int8_t sign_root;
	};
	struct ImgImageState
	{
		uint2* ranges;
	#if IMG_ACCUM_MODE == 0
	//只在平均数下使用
	    int* accum_cnt;
	#endif
		ImgHit* hits;   // 大小 = img_width * img_height * IMG_MAX_HITS
		static ImgImageState fromChunk(char*& chunk, size_t N1, size_t N2);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	template<typename T> 
	size_t required(size_t P1, size_t P2)
	{
		char* size = nullptr;
		T::fromChunk(size, P1, P2);
		return ((size_t)size) + 128;
	}
};
