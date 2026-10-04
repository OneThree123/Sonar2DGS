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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 1 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16
#define TX_TOPK 100
#define TX_THRESH 1E-3F
#define OTHERS_NUM 7
#ifndef IMG_ACCUM_MODE
#define IMG_ACCUM_MODE 1   // 0: mean, 1: sum, 2: logsumexp
#endif
constexpr int MAX_AZIMUTH_BINS = 1024;//512;
constexpr int MAX_RANGE_BINS   = 1024;//666;
constexpr float GS_CUTOFF = 3.0f;
//图像变换中需要的参数
enum
{
    IMG_OX = 0,        // o_s.x
    IMG_OY = 1,        // o_s.y
    IMG_RUX = 2,       // ru_s.x   (高斯局部u轴，单位向量，在声呐坐标系下)
    IMG_RUY = 3,       // ru_s.y
    IMG_RVX = 4,       // rv_s.x   (高斯局部v轴，单位向量，在声呐坐标系下)
    IMG_RVY = 5,       // rv_s.y
    IMG_OU = 6,        // dot(ru_s, o_s)  物理长度坐标
    IMG_OV = 7,        // dot(rv_s, o_s)
    IMG_H  = 8,        // dot(n_s , o_s)  平面法向距离
    IMG_INV_SU = 9,    // 1 / (scale_u * modifier)
    IMG_INV_SV = 10,   // 1 / (scale_v * modifier)
    SONAR_IMG_PARAM_STRIDE = 11
};
#endif