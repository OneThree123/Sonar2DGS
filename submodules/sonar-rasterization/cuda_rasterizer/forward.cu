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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __constant__ float2 f_theta_dir[MAX_AZIMUTH_BINS];
__device__ __constant__ float  f_range_val[MAX_RANGE_BINS];

#ifndef DEBUG_PREPROCESS_REPROJECT
#define DEBUG_PREPROCESS_REPROJECT 0
#endif

#if DEBUG_PREPROCESS_REPROJECT
__device__ int g_preprocess_debug_printed = 0;
#endif

// 将三角函数常量从主机复制到设备常量内存
void copyConstants2Forward(float2* d_theta_dir, float* d_range_val, float num_theta, float num_range)
{
    cudaMemcpyToSymbol(f_theta_dir, d_theta_dir, num_theta * sizeof(float2));
    cudaMemcpyToSymbol(f_range_val, d_range_val, num_range * sizeof(float));
}

//viewmatrix理解为行主序的左乘行向量运算矩阵
__device__ __forceinline__ glm::vec3 rotate_world_to_sonar(
    const glm::vec3& v_w,
    const float* viewmatrix)
{
    return glm::vec3(
        viewmatrix[0] * v_w.x + viewmatrix[4] * v_w.y + viewmatrix[8]  * v_w.z,
        viewmatrix[1] * v_w.x + viewmatrix[5] * v_w.y + viewmatrix[9]  * v_w.z,
        viewmatrix[2] * v_w.x + viewmatrix[6] * v_w.y + viewmatrix[10] * v_w.z
    );
}

//可见性判断
__device__ bool in_sonar_fov(
	const glm::vec3& gs_xyz_w,
	const glm::vec3& cam_pos_w,
	const float* viewmatrix, 
    const float max_range,
    const float tan_fovx,
    const float tan_fovy,
	glm::vec3& s2gs_trans_w, 
	glm::vec3& s2gs_trans_s, 
	float& gs_range_s
)
{
	//世界坐标系下，相机指向高斯的向量
	s2gs_trans_w = gs_xyz_w - cam_pos_w;
    gs_range_s = glm::length(s2gs_trans_w);
    // 1) 距离门限
    if (gs_range_s < 0.1f || gs_range_s > max_range) return false;

    // 2) 航向/俯仰门限（等价于角度阈值，避免atan）
    // yaw:   |y/x| <= tan_fovx
    // pitch: |z/x| <= tan_fovy
	//声呐坐标系下，相机指向高斯的向量
	s2gs_trans_s = rotate_world_to_sonar(s2gs_trans_w, viewmatrix);
    const float yaw_ratio   = fabsf(s2gs_trans_s.y / s2gs_trans_s.x);
    const float pitch_ratio = fabsf(s2gs_trans_s.z / s2gs_trans_s.x);

    return (yaw_ratio <= tan_fovx) && (pitch_ratio <= tan_fovy);

}

// 高斯坐标求透射坐标
__device__ __forceinline__ bool project_gs2tx(
    const glm::mat3& T,
    const float u,
    const float v,
    float2& pix)
{
    const float w = T[2][0] * u + T[2][1] * v + T[2][2];
    if (fabsf(w) < 1e-8f)
        return false;

    const float inv_w = 1.0f / w;
    pix.x = (T[0][0] * u + T[0][1] * v + T[0][2]) * inv_w;
    pix.y = (T[1][0] * u + T[1][1] * v + T[1][2]) * inv_w;

    return true;
}
//gs坐标求像素坐标
__device__ __forceinline__ bool project_gs2pix(
    const glm::vec3& o_s,
    const glm::vec3& ru_s,
    const glm::vec3& rv_s,
    const float su,
    const float sv,
    const float u,      // 高斯标准局部坐标
    const float v,
    const float theta_min,
    const float inv_dtheta,
    const float range_min,
    const float inv_dr,
    float2& pix)
{
    // 注意：这里先把标准坐标(u,v)恢复成物理平面位移
    glm::vec3 p_s = o_s + (u * su) * ru_s + (v * sv) * rv_s;

    const float r = glm::length(p_s);
    if (r <= 1e-8f) return false;

    float th = atan2f(p_s.y, p_s.x);

    pix.x = (th - theta_min) * inv_dtheta - 0.5f; // theta -> x
    pix.y = (r  - range_min) * inv_dr     - 0.5f; // range -> y
    return true;
}
#if DEBUG_PREPROCESS_REPROJECT
__device__ __forceinline__ bool project_gs2pix(
    const float* params,
    const float u,
    const float v,
    const float theta_min,
    const float inv_dtheta,
    const float range_min,
    const float inv_dr,
    float2& pix)
{
    const float inv_su = params[IMG_INV_SU];
    const float inv_sv = params[IMG_INV_SV];
    if (fabsf(inv_su) < 1e-8f || fabsf(inv_sv) < 1e-8f) return false;

    const float u_phys = u / inv_su;
    const float v_phys = v / inv_sv;

    const float px = params[IMG_OX] + u_phys * params[IMG_RUX] + v_phys * params[IMG_RVX];
    const float py = params[IMG_OY] + u_phys * params[IMG_RUY] + v_phys * params[IMG_RVY];

    const float pu = params[IMG_OU] + u_phys;
    const float pv = params[IMG_OV] + v_phys;
    const float h  = params[IMG_H];
    const float r = sqrtf(pu * pu + pv * pv + h * h);
    if (r <= 1e-8f) return false;

    const float th = atan2f(py, px);
    pix.x = (th - theta_min) * inv_dtheta - 0.5f;
    pix.y = (r  - range_min) * inv_dr     - 0.5f;
    return true;
}
#endif

/*图像方向*/
//图像坐标转高斯坐标，核心是解一个二次方程，可能有0/1/2个解
__device__ __forceinline__ int imgPixToGs(
    const int th_bin,
    const int r_bin,
    const float* params,          // img_params + idx * SONAR_IMG_PARAM_STRIDE
    const float cutoff,
    float2 out_uv[2]              // 输出标准高斯坐标
)
{
    const float c = f_theta_dir[th_bin].x;
    const float s = f_theta_dir[th_bin].y;
    const float r = f_range_val[r_bin];

    const float ox = params[IMG_OX];
    const float oy = params[IMG_OY];
    const float rux = params[IMG_RUX];
    const float ruy = params[IMG_RUY];
    const float rvx = params[IMG_RVX];
    const float rvy = params[IMG_RVY];
    const float ou  = params[IMG_OU];
    const float ov  = params[IMG_OV];
    const float h   = params[IMG_H];
    const float inv_su = params[IMG_INV_SU];
    const float inv_sv = params[IMG_INV_SV];

    const float alpha = -s * rux + c * ruy;
    const float beta  = -s * rvx + c * rvy;
    const float gamma = -s * ox  + c * oy;

    const float rho2 = r * r - h * h;
    if (rho2 < 0.0f)
    {
        out_uv[0] = make_float2(0.0f, rho2);
        out_uv[1] = make_float2(r, h);
        return 0;
    } 

    const float d2 = alpha * alpha + beta * beta;
    if (d2 < 1e-10f) 
    {
        out_uv[0] = make_float2(1.0f, d2);
        return 0;
    } 

    const float q = alpha * ou + beta * ov - gamma;
    float delta = d2 * rho2 - q * q;
    if (delta < 0.0f)
    {
        if (delta > -1e-6f) delta = 0.0f;
        else 
        {
            out_uv[0] = make_float2(2.0f, delta);
            return 0;
        } 
    }

    const float root = sqrtf(delta);
    const float inv_d2 = 1.0f / d2;

    int nsol = 0;

    // root #0
    {
        const float u_phys = q * alpha * inv_d2 - ou - beta  * root * inv_d2;
        const float v_phys = q * beta  * inv_d2 - ov + alpha * root * inv_d2;

        const float u_std = u_phys * inv_su;
        const float v_std = v_phys * inv_sv;

        if (u_std * u_std + v_std * v_std <= cutoff * cutoff)
        {
            out_uv[nsol++] = make_float2(u_std, v_std);
        }
    }

    // root #1
    if (root > 1e-7f)
    {
        const float u_phys = q * alpha * inv_d2 - ou + beta  * root * inv_d2;
        const float v_phys = q * beta  * inv_d2 - ov - alpha * root * inv_d2;

        const float u_std = u_phys * inv_su;
        const float v_std = v_phys * inv_sv;

        if (u_std * u_std + v_std * v_std <= cutoff * cutoff)
        {
            out_uv[nsol++] = make_float2(u_std, v_std);
        }
    }

    return nsol;
}

//从高斯坐标变化为 tx 坐标
__device__ __forceinline__ bool project_Gs2Tx(
    const float* T9,
    const float u,
    const float v,
    float2& pix)
{
    const float w = T9[6] * u + T9[7] * v + T9[8];
    if (fabsf(w) < 1e-8f) return false;

    const float inv_w = 1.0f / w;
    pix.x = (T9[0] * u + T9[1] * v + T9[2]) * inv_w;
    pix.y = (T9[3] * u + T9[4] * v + T9[5]) * inv_w;
    return true;
}
// 声呐漫反射强度计算
__device__ __forceinline__ bool compute_normal_and_intensity(
    const glm::vec4& gs_quat,
    const glm::vec3& s2gs_trans,
    const float gs_reflection,
	glm::mat3& gs_rot,
    float3& out_normal,
    float& out_intensity)
{
    gs_rot = quat_to_rotmat(gs_quat);//？？是否遵循列主序
    glm::vec3 gs_normal = gs_rot[2];//？？列主序下是否合理，列主序拿第三列应该合理吧？
    out_normal = make_float3(gs_normal.x, gs_normal.y, gs_normal.z);

    glm::vec3 s2gs_dir = glm::normalize(s2gs_trans);
    const float cos_term = glm::dot(s2gs_dir, gs_normal);
    const float front_cos = -cos_term;
    if (front_cos < 0.0f)
    {
        out_intensity = 0.0f;
        return false;
    }

    out_intensity = gs_reflection * front_cos;
    return true;
}
// 计算 tx 参数：矩阵、图像中心、半径、深度、tiles 数量
__device__ __forceinline__ bool compute_tx_params(
    const glm::vec3& gs_pos,
	    const glm::vec2& gs_scale,
	    const float gs_scale_modifier,
	    const float lowpass_variance,
	    const glm::mat3& gs_rot,
    const float gs_range_s,
    const float* projmatrix,
    const int W,
    const int H,
    const dim3 grid,

		    float* tx_mat9,          // 指向 tx2gs_transMats + 9 * idx
		    float4& tx_bound,        // tx_bounds[idx]，{xmin, xmax, ymin, ymax}
		    float2& tx_center_2d,
		    float& tx_depth,         // tx_depths[idx]
	    uint32_t& tx_tiles_touched,
    float& tx_raddi,

    const float cutoff = 3.0f
)
{
    // =========================================================
    // 1. 计算变换矩阵 T（等价于 compute_transmat）
    // =========================================================
	    glm::mat3 S = scale_to_mat(gs_scale, gs_scale_modifier);
	    glm::mat3 L = gs_rot * S;
	    // glm::mat3 L = glm::transpose(gs_rot)  * S;

    glm::mat3x4 splat2world = glm::mat3x4(//？？列主序取前两列应该合理，这里是3列4行？
        glm::vec4(L[0], 0.0f),
        glm::vec4(L[1], 0.0f),
        glm::vec4(gs_pos, 1.0f)
    );

    glm::mat4 world2ndc = glm::mat4(
        projmatrix[0],  projmatrix[4],  projmatrix[8],  projmatrix[12],
        projmatrix[1],  projmatrix[5],  projmatrix[9],  projmatrix[13],
        projmatrix[2],  projmatrix[6],  projmatrix[10], projmatrix[14],
        projmatrix[3],  projmatrix[7],  projmatrix[11], projmatrix[15]
    );

    glm::mat3x4 ndc2pix = glm::mat3x4(//？？需要检查是否合理
        glm::vec4(float(W) * 0.5f, 0.0f, 0.0f, float(W - 1) * 0.5f),
        glm::vec4(0.0f, float(H) * 0.5f, 0.0f, float(H - 1) * 0.5f),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );

    glm::mat3 T = glm::transpose(splat2world) * world2ndc * ndc2pix;//？？这里为什么对splat2world取转置，应该不需要吧

    // 按你当前的存储方式写回：每个高斯 9 个 float，按 T[0], T[1], T[2] 三列存
    tx_mat9[0] = T[0][0];
    tx_mat9[1] = T[0][1];
    tx_mat9[2] = T[0][2];

    tx_mat9[3] = T[1][0];
    tx_mat9[4] = T[1][1];
    tx_mat9[5] = T[1][2];

    tx_mat9[6] = T[2][0];
    tx_mat9[7] = T[2][1];
    tx_mat9[8] = T[2][2];

    // =========================================================
    // 2. 投影高斯局部坐标系中轴与 cutoff 边界的四个交点
    //    四个点分别是：
    //    (+cutoff, 0), (-cutoff, 0), (0, +cutoff), (0, -cutoff)
    // =========================================================
    float2 p0, p1, p2, p3;
    if (!project_gs2tx(T,  cutoff, 0.0f, p0)) return false;
    if (!project_gs2tx(T, -cutoff, 0.0f, p1)) return false;
    if (!project_gs2tx(T, 0.0f,  cutoff, p2)) return false;
    if (!project_gs2tx(T, 0.0f, -cutoff, p3)) return false;

	    if (!project_gs2tx(T, 0.0f, 0.0f, tx_center_2d)) return false;
	    const float lowpass_radius = cutoff * sqrtf(fmaxf(lowpass_variance, 1e-8f));

	    float xmin = fminf(
	        fminf(fminf(p0.x, p1.x), fminf(p2.x, p3.x)),
	        tx_center_2d.x - lowpass_radius);
	    float xmax = fmaxf(
	        fmaxf(fmaxf(p0.x, p1.x), fmaxf(p2.x, p3.x)),
	        tx_center_2d.x + lowpass_radius);
	    float ymin = fminf(
	        fminf(fminf(p0.y, p1.y), fminf(p2.y, p3.y)),
	        tx_center_2d.y - lowpass_radius);
	    float ymax = fmaxf(
	        fmaxf(fmaxf(p0.y, p1.y), fmaxf(p2.y, p3.y)),
	        tx_center_2d.y + lowpass_radius);
	tx_bound = make_float4(xmin, xmax, ymin, ymax);

    const float dx_u = p0.x - p1.x;
    const float dy_u = p0.y - p1.y;
    const float dx_v = p2.x - p3.x;
    const float dy_v = p2.y - p3.y;

    tx_raddi = dx_u * dx_u + dy_u * dy_u + dx_v * dx_v + dy_v * dy_v;

    // =========================================================
    // 3. 根据边界算 tile 覆盖范围
    //    这里 rect_max_* 用“开区间”
    // =========================================================
    int rect_min_x = max(0, (int)floorf(xmin / BLOCK_X));
    int rect_min_y = max(0, (int)floorf(ymin / BLOCK_Y));

    int rect_max_x = min((int)grid.x, max(0, (int)floorf(xmax / BLOCK_X) + 1));
    int rect_max_y = min((int)grid.y, max(0, (int)floorf(ymax / BLOCK_Y) + 1));

    if (rect_max_x <= rect_min_x || rect_max_y <= rect_min_y)
        return false;

    tx_tiles_touched = (uint32_t)((rect_max_x - rect_min_x) * (rect_max_y - rect_min_y));
    tx_depth = gs_range_s;

    return true;
}
__device__ __forceinline__ bool compute_img_params(
    const glm::vec3& o_s,   // 声呐到高斯中心的声呐系向量
	    const glm::vec2& gs_scale,
	    const float gs_scale_modifier,
	    const float lowpass_variance,
	    const glm::mat3& gs_rot,         // 世界系下高斯旋转
    const float* viewmatrix,

    // 声呐图像采样参数
    const float theta_min,
    const float inv_dtheta,
    const float range_min,
    const float inv_dr,

    const dim3 img_grid,

    float* img_params,              // img_transparas + idx * SONAR_IMG_PARAM_STRIDE
    float4& img_bound,               // {xmin, xmax, ymin, ymax}，在img像素坐标里
    uint32_t& img_tiles_touched,

    const float cutoff = 3.0f
)
{
    // ---------------------------------------------------------
    // 1) 先把中心、局部轴、法向都转到声呐坐标系
    // ---------------------------------------------------------
    const float su = gs_scale.x * gs_scale_modifier;
    const float sv = gs_scale.y * gs_scale_modifier;
    if (su <= 1e-8f || sv <= 1e-8f)
        return false;

    const glm::vec3 ru_s = rotate_world_to_sonar(gs_rot[0], viewmatrix); // 单位u轴
    const glm::vec3 rv_s = rotate_world_to_sonar(gs_rot[1], viewmatrix); // 单位v轴
    const glm::vec3 n_s  = rotate_world_to_sonar(gs_rot[2], viewmatrix); // 单位法向

    const float ou = glm::dot(ru_s, o_s);
    const float ov = glm::dot(rv_s, o_s);
    const float h  = glm::dot(n_s,  o_s);

    // ---------------------------------------------------------
    // 2) 存后续 render 反解 pix->gs 所需参数
    // ---------------------------------------------------------
    img_params[IMG_OX]        = o_s.x;
    img_params[IMG_OY]        = o_s.y;
    img_params[IMG_RUX]       = ru_s.x;
    img_params[IMG_RUY]       = ru_s.y;
    img_params[IMG_RVX]       = rv_s.x;
    img_params[IMG_RVY]       = rv_s.y;
    img_params[IMG_OU]        = ou;
    img_params[IMG_OV]        = ov;
    img_params[IMG_H]         = h;
    img_params[IMG_INV_SU]    = 1.0f / su;
    img_params[IMG_INV_SV]    = 1.0f / sv;
    // ---------------------------------------------------------
    // 3) 用4个cutoff边界点做 img 投影边界
    //    与 compute_tx_params 的思路保持一致
    // ---------------------------------------------------------
    float2 p0, p1, p2, p3;
    if (!project_gs2pix(
            o_s, ru_s, rv_s, su, sv,
            cutoff, 0.0f,
            theta_min, inv_dtheta, range_min, inv_dr, p0)) return false;

    if (!project_gs2pix(
            o_s, ru_s, rv_s, su, sv,
           -cutoff, 0.0f,
            theta_min, inv_dtheta, range_min, inv_dr, p1)) return false;

    if (!project_gs2pix(
            o_s, ru_s, rv_s, su, sv,
            0.0f, cutoff,
            theta_min, inv_dtheta, range_min, inv_dr, p2)) return false;

    if (!project_gs2pix(
            o_s, ru_s, rv_s, su, sv,
            0.0f,-cutoff,
            theta_min, inv_dtheta, range_min, inv_dr, p3)) return false;

	    const float intrinsic_xmin = fminf(fminf(p0.x, p1.x), fminf(p2.x, p3.x));
	    const float intrinsic_xmax = fmaxf(fmaxf(p0.x, p1.x), fmaxf(p2.x, p3.x));
	    const float intrinsic_ymin = fminf(fminf(p0.y, p1.y), fminf(p2.y, p3.y));
	    const float intrinsic_ymax = fmaxf(fmaxf(p0.y, p1.y), fmaxf(p2.y, p3.y));
	    const float center_x = 0.5f * (intrinsic_xmin + intrinsic_xmax);
	    const float center_y = 0.5f * (intrinsic_ymin + intrinsic_ymax);
	    const float lowpass_radius = cutoff * sqrtf(fmaxf(lowpass_variance, 1e-8f));
	    const float xmin = fminf(intrinsic_xmin, center_x - lowpass_radius);
	    const float xmax = fmaxf(intrinsic_xmax, center_x + lowpass_radius);
	    const float ymin = fminf(intrinsic_ymin, center_y - lowpass_radius);
	    const float ymax = fmaxf(intrinsic_ymax, center_y + lowpass_radius);

    img_bound = make_float4(xmin, xmax, ymin, ymax);

    // ---------------------------------------------------------
    // 4) 统计 img tile 覆盖数量
    // ---------------------------------------------------------
    const int rect_min_x = max(0, (int)floorf(xmin / BLOCK_X));
    const int rect_min_y = max(0, (int)floorf(ymin / BLOCK_Y));

    const int rect_max_x = min((int)img_grid.x, max(0, (int)floorf(xmax / BLOCK_X) + 1));
    const int rect_max_y = min((int)img_grid.y, max(0, (int)floorf(ymax / BLOCK_Y) + 1));

    if (rect_max_x <= rect_min_x || rect_max_y <= rect_min_y)
        return false;

    img_tiles_touched = (uint32_t)((rect_max_x - rect_min_x) * (rect_max_y - rect_min_y));
    return true;
}
// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
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
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	//初始化为0，可以使不通过可见性的高斯无法参与排序和渲染
	tx_bounds[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
	tx_tiles_touched[idx] = 0;
	img_bounds[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
	img_tiles_touched[idx] = 0;
    //r_raddi<0表示不可见
    r_raddi[idx] = -1.0f;

	/*判断可见性*/
	//世界坐标指向向量
	glm::vec3 os_s;
	glm::vec3 s2gs_trans;
	//高斯到声呐坐标距离
	float gs_range_s;
	//可见性判断，同时计算法向量和距离
	if (!in_sonar_fov(gs_trans[idx], (*cam_pos), viewmatrix, 
					range_max, tan_fovx, tan_fovy, 
					s2gs_trans, os_s, gs_range_s))
		return;

	/*计算中立量：
	法向量和透明度*/
	glm::mat3 gs_rot;
	if (!compute_normal_and_intensity(
		gs_quats[idx],
		s2gs_trans,
		gs_reflections[idx],
		gs_rot,
		gs_normals[idx],
		gs_intensities[idx]
	))
	{
		return;
	}
	
	/*计算透射相关参数：
	变换矩阵、投影边界、tiles数量、深度
	*/
	float tx_raddi = 0.0f;
	if (!compute_tx_params(
			gs_trans[idx],
				gs_scales[idx],
				gs_scale_modifier,
				lowpass_variance,
				gs_rot,
			gs_range_s,
			projmatrix,
			tx_width, tx_height,
			tx_grid,
					tx_transMats + 9 * idx,
					tx_bounds[idx],
					tx_centers_2d[idx],
					tx_depths[idx],
				tx_tiles_touched[idx],
	            tx_raddi))
	{
		return;
	}

    const float su = gs_scales[idx].x * gs_scale_modifier;
    const float sv = gs_scales[idx].y * gs_scale_modifier;
    const glm::vec3 axis_u = gs_rot[0] * su;
    const glm::vec3 axis_v = gs_rot[1] * sv;
    tx_centers[idx] = make_float3(gs_trans[idx].x, gs_trans[idx].y, gs_trans[idx].z);
    tx_axis_u[idx] = make_float3(axis_u.x, axis_u.y, axis_u.z);
    tx_axis_v[idx] = make_float3(axis_v.x, axis_v.y, axis_v.z);



	/*计算声呐图像相关参数
	变换矩阵、投影边界、tiles数量
	*/
	if (!compute_img_params(
			os_s,                     
				gs_scales[idx],
				gs_scale_modifier,
				lowpass_variance,
				gs_rot,
			viewmatrix,
			theta_min,
			inv_dtheta,
			range_min,
			inv_dr,
			img_grid,
			img_params + SONAR_IMG_PARAM_STRIDE * idx,
			img_bounds[idx],
			img_tiles_touched[idx]))
	{
		return;
	}

	// gs_raddi is defined in the final sonar image coordinate system.
	// Store the sum of squares of the image-space bounding-box coordinates:
	// xmin^2 + xmax^2 + ymin^2 + ymax^2.
	const float4 image_bound = img_bounds[idx];
	r_raddi[idx] =
		image_bound.x * image_bound.x + image_bound.y * image_bound.y +
		image_bound.z * image_bound.z + image_bound.w * image_bound.w;

#if DEBUG_PREPROCESS_REPROJECT
    {
        const float2 gs1 = make_float2(0.0f, 0.0f);
        const float* T9 = tx_transMats + 9 * idx;
        const float* ip = img_params + SONAR_IMG_PARAM_STRIDE * idx;

        const glm::mat3 T_dbg(
            glm::vec3(T9[0], T9[1], T9[2]),
            glm::vec3(T9[3], T9[4], T9[5]),
            glm::vec3(T9[6], T9[7], T9[8]));

        float2 tx1 = make_float2(-9999.0f, -9999.0f);
        float2 img1 = make_float2(-9999.0f, -9999.0f);

        const bool ok_tx1 = project_gs2tx(T_dbg, gs1.x, gs1.y, tx1);
        const bool ok_img1 = project_gs2pix(ip, gs1.x, gs1.y, theta_min, inv_dtheta, range_min, inv_dr, img1);

        int th_bin = 0;
        int r_bin = 0;
        if (ok_img1)
        {
            th_bin = max(0, min(MAX_AZIMUTH_BINS - 1, (int)floorf(img1.x + 0.5f)));
            r_bin  = max(0, min(MAX_RANGE_BINS - 1,   (int)floorf(img1.y + 0.5f)));
        }

        float2 uv_sol[2]= {
            make_float2(-9999.0f, -9999.0f),
            make_float2(-9999.0f, -9999.0f)
        };
        const int nsol = ok_img1 ? imgPixToGs(th_bin, r_bin, ip, GS_CUTOFF, uv_sol) : 0;

        float2 tx2_sol[2] = {
            make_float2(-9999.0f, -9999.0f),
            make_float2(-9999.0f, -9999.0f)
        };
        bool ok_tx2_sol[2] = {false, false};
        for (int s = 0; s < nsol; ++s)
        {
            ok_tx2_sol[s] = project_gs2tx(T_dbg, uv_sol[s].x, uv_sol[s].y, tx2_sol[s]);
        }

        if (atomicCAS(&g_preprocess_debug_printed, 0, 1) == 0)
        {
            printf("\n\n[preprocess dbg] idx=%u ok_tx1=%d ok_img1=%d nsol=%d th_bin=%d r_bin=%d\n",
                   (unsigned int)idx, (int)ok_tx1, (int)ok_img1, nsol, th_bin, r_bin);

            printf("gs1=(%.9g %.9g)\n tx1=(%.9g %.9g)\n img1=(%.9g %.9g)\n",
                   gs1.x, gs1.y, tx1.x, tx1.y, img1.x, img1.y);
            if (nsol <= 0)
            {
                printf("gs2={}\n");
                printf("(errorTyp,  errorVal0)=(%.9g %.9g)\n",
                    uv_sol[0].x, uv_sol[0].y);
                printf("(errorVal1, errorVal2)=(%.9g %.9g)\n",
                    uv_sol[1].x, uv_sol[1].y);
            }
            else if (nsol == 1)
            {
                printf("gs2={(%.9g %.9g)}\n",
                       uv_sol[0].x, uv_sol[0].y);
            }
            else
            {
                printf("gs2={(%.9g %.9g), (%.9g %.9g)}\n",
                       uv_sol[0].x, uv_sol[0].y,
                       uv_sol[1].x, uv_sol[1].y);
            }

            const int ntx2 = (int)ok_tx2_sol[0] + (int)ok_tx2_sol[1];
            if (ntx2 <= 0)
            {
                printf("tx2={}\n\n");
            }
            else if (ntx2 == 1)
            {
                const int s = ok_tx2_sol[0] ? 0 : 1;
                printf("tx2={(%.9g %.9g)}\n\n",
                       tx2_sol[s].x, tx2_sol[s].y);
            }
            else
            {
                printf("tx2={(%.9g %.9g), (%.9g %.9g)}\n\n",
                       tx2_sol[0].x, tx2_sol[0].y,
                       tx2_sol[1].x, tx2_sol[1].y);
            }

            printf("gs_pos_w=(%.9g %.9g %.9g) quat=(%.9g %.9g %.9g %.9g) scale=(%.9g %.9g)\n",
                   gs_trans[idx].x, gs_trans[idx].y, gs_trans[idx].z,
                   gs_quats[idx].x, gs_quats[idx].y, gs_quats[idx].z, gs_quats[idx].w,
                   gs_scales[idx].x, gs_scales[idx].y);

            printf("cam_pos_w=(%.9g %.9g %.9g) os_s=(%.9g %.9g %.9g)\n\n",
                   cam_pos->x, cam_pos->y, cam_pos->z, os_s.x, os_s.y, os_s.z);
        }

    }
#endif


}
/*透射方向*/
//透射坐标转高斯坐标
__forceinline__ __device__ bool txPixToGs(
    int px, int py,
    const float* T9,      // 9 floats
	    float2& uv)
{
    float3 Tu = {T9[0], T9[1], T9[2]};
    float3 Tv = {T9[3], T9[4], T9[5]};
    float3 Tw = {T9[6], T9[7], T9[8]};

    float3 k = px * Tw - Tu;
    float3 l = py * Tw - Tv;
    float3 p = cross(k, l);
    if (fabsf(p.z) < 1e-8f) return false;

    uv.x = p.x / p.z;
    uv.y = p.y / p.z;
	return true;
}

__device__ __forceinline__ float dot3(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 point_on_splat_world(
    const float3& center,
    const float3& axis_u,
    const float3& axis_v,
    const float2& uv)
{
    return make_float3(
        center.x + uv.x * axis_u.x + uv.y * axis_v.x,
        center.y + uv.x * axis_u.y + uv.y * axis_v.y,
        center.z + uv.x * axis_u.z + uv.y * axis_v.z);
}

__device__ __forceinline__ float range_from_cam(
    const float3& point,
    const glm::vec3& cam_pos)
{
    const float3 ray = make_float3(
        point.x - cam_pos.x,
        point.y - cam_pos.y,
        point.z - cam_pos.z);
    return sqrtf(fmaxf(dot3(ray, ray), 1e-12f));
}

//透射方向渲染函数
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderTxCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float range_max,
	    const glm::vec3* __restrict__ cam_pos,
	    const float* __restrict__ tx_transMats,
	    const float2* __restrict__ tx_centers_2d,
	    const float3* __restrict__ tx_centers,
    const float3* __restrict__ tx_axis_u,
    const float3* __restrict__ tx_axis_v,
	    const float3* __restrict__ gs_normals,
	    const float* __restrict__ opacity,
	    const float lowpass_variance,
	    uint32_t* __restrict__ topk_ids,
    float* __restrict__ topk_vis_weight,
    uint8_t* __restrict__ topk_count,
	float* __restrict__ out_others)
{
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;

    bool inside = pix.x < W && pix.y < H;
    bool done   = !inside;

    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int toDo = range.y - range.x;

    __shared__ uint32_t sh_id[BLOCK_SIZE];
    __shared__ float   sh_opacity[BLOCK_SIZE];
	    __shared__ float3   sh_Tu[BLOCK_SIZE];
	    __shared__ float3   sh_Tv[BLOCK_SIZE];
	    __shared__ float3   sh_Tw[BLOCK_SIZE];
	    __shared__ float2 sh_center_2d[BLOCK_SIZE];
	    __shared__ float3 sh_center[BLOCK_SIZE];
    __shared__ float3 sh_axis_u[BLOCK_SIZE];
    __shared__ float3 sh_axis_v[BLOCK_SIZE];
    __shared__ float3 sh_normal[BLOCK_SIZE];

    float T = 1.0f;
    float T_before = 1.0f;
    uint8_t count = 0;
    //others
    float depth = 0.0f;
    float aver_depth = 0.0f;
    float3 normal = {0.0f, 0.0f, 0.0f};
    float median_depth = 0.0f;
    float distortion = 0.0f;
    float m = 0.0f;
    float prefix_w = 0.0f;
    float prefix_wm = 0.0f;
    float prefix_wm2 = 0.0f;

    for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
    {
        if (__syncthreads_count(done) == BLOCK_SIZE) break;

        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            uint32_t gid = point_list[range.x + progress];
            sh_id[block.thread_rank()] = gid;
            sh_opacity[block.thread_rank()] = opacity[gid];
	            sh_Tu[block.thread_rank()] = {tx_transMats[9*gid+0], tx_transMats[9*gid+1], tx_transMats[9*gid+2]};
            sh_Tv[block.thread_rank()] = {tx_transMats[9*gid+3], tx_transMats[9*gid+4], tx_transMats[9*gid+5]};
	            sh_Tw[block.thread_rank()] = {tx_transMats[9*gid+6], tx_transMats[9*gid+7], tx_transMats[9*gid+8]};
	            sh_center_2d[block.thread_rank()] = tx_centers_2d[gid];
	            sh_center[block.thread_rank()] = tx_centers[gid];
            sh_axis_u[block.thread_rank()] = tx_axis_u[gid];
            sh_axis_v[block.thread_rank()] = tx_axis_v[gid];
            sh_normal[block.thread_rank()] = gs_normals[gid];
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
        {

            float T9[9] = {//？？这里和后续函数txPixToGs用glm库，统一风格
                sh_Tu[j].x, sh_Tu[j].y, sh_Tu[j].z,
                sh_Tv[j].x, sh_Tv[j].y, sh_Tv[j].z,
                sh_Tw[j].x, sh_Tw[j].y, sh_Tw[j].z
            };

            float2 uv;
            if (!txPixToGs(pix.x, pix.y, T9, uv)) continue;

	            const float intrinsic_rho = uv.x * uv.x + uv.y * uv.y;
	            const float dx = (float)pix.x - sh_center_2d[j].x;
	            const float dy = (float)pix.y - sh_center_2d[j].y;
	            const float lowpass_rho =
	                (dx * dx + dy * dy) / fmaxf(lowpass_variance, 1e-8f);
	            const float rho = fminf(intrinsic_rho, lowpass_rho);
            float power = -0.5f * rho;

            float opa = sh_opacity[j];
            float alpha = min(0.99f, opa * expf(power));
            if (alpha < 1.0f / 255.0f) continue;

            // 当前“已保留高斯集合”下的贡献
            float vis_weight = alpha * T;

            // 太小就直接忽略：
            // 既不写 topk，也不参与 others，也不衰减 T
            if (vis_weight < TX_THRESH)
                continue;

            // 记录到 topk
            topk_ids[pix_id * TX_TOPK + count] = sh_id[j];
            topk_vis_weight[pix_id * TX_TOPK + count] = vis_weight;
            count++;

            // others 也只由保留下来的项计算
            const float3 hit_point = point_on_splat_world(
                sh_center[j], sh_axis_u[j], sh_axis_v[j], uv);
            depth = range_from_cam(hit_point, *cam_pos);
            normal.x += vis_weight * sh_normal[j].x;
            normal.y += vis_weight * sh_normal[j].y;
            normal.z += vis_weight * sh_normal[j].z;
            aver_depth += depth * vis_weight;

            m = fminf(fmaxf(depth / range_max, 0.f), 1.f);
            // Order-independent squared depth distortion:
            //   sum_{j<i} w_i w_j (m_i - m_j)^2
            // = w_i (m_i^2 sum(w_j) + sum(w_j m_j^2)
            //          - 2 m_i sum(w_j m_j)).
            // Unlike the signed linear moment form, this remains non-negative
            // when center-depth sorting and per-pixel hit-depth sorting differ.
            distortion += vis_weight * (
                m * m * prefix_w + prefix_wm2 - 2.0f * m * prefix_wm);
            prefix_w += vis_weight;
            prefix_wm += m * vis_weight;
            prefix_wm2 += m * m * vis_weight;

            // 只有保留下来的项才更新透射率
            T_before = T;
            T *= (1.0f - alpha);

            // 到这里 topk / others / T 语义完全一致
            if ((T < TX_THRESH) || (count >= TX_TOPK))
            {
                done = true;
                break;
            }
        }
    }

    if (inside)
    {
        topk_count[pix_id] = count;
        if (out_others)
        {
            int picture_length = H * W;
            out_others[pix_id + DEPTH_OFFSET * picture_length] = aver_depth;
            out_others[pix_id + ALPHA_OFFSET * picture_length] = 1.0f - T;
            out_others[pix_id + NORMAL_OFFSET * picture_length] = normal.x;
            out_others[pix_id + (NORMAL_OFFSET + 1) * picture_length] = normal.y;
            out_others[pix_id + (NORMAL_OFFSET + 2) * picture_length] = normal.z;
            out_others[pix_id + DISTORTION_OFFSET * picture_length] = distortion;
        }
    }

}

//从 tx 像素缓存里查某个高斯的透明度
__device__ __forceinline__ float lookupTxVis(
    const int tx_x,
    const int tx_y,
    const uint32_t gid,
    const int W,
    const int H,
    const uint32_t* tx_topk_ids,
    const float* tx_topk_vis,
    const uint8_t* tx_topk_count)
{
    if (tx_x < 0 || tx_x >= W || tx_y < 0 || tx_y >= H) return 0.0f;

    const int pix_id = tx_y * W + tx_x;
    const int cnt = (int)tx_topk_count[pix_id];
    const int base = pix_id * TX_TOPK;

    for (int k = 0; k < cnt; ++k)
    {
        if (tx_topk_ids[base + k] == gid)
            return tx_topk_vis[base + k];
    }
    return 0.0f;
}
//查询透明度，插值版
__device__ __forceinline__ float lookupTxVisBilinear(
    const float tx_x,
    const float tx_y,
    const uint32_t gid,
    const int W,
    const int H,
    const uint32_t* tx_topk_ids,
    const float* tx_topk_vis,
    const uint8_t* tx_topk_count)
{
    const int x0 = (int)floorf(tx_x);
    const int y0 = (int)floorf(tx_y);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = tx_x - x0;
    const float fy = tx_y - y0;

    float acc = 0.0f;

    acc += (1.0f - fx) * (1.0f - fy) *
        lookupTxVis(x0, y0, gid, W, H, tx_topk_ids, tx_topk_vis, tx_topk_count);

    acc += fx * (1.0f - fy) *
        lookupTxVis(x1, y0, gid, W, H, tx_topk_ids, tx_topk_vis, tx_topk_count);

    acc += (1.0f - fx) * fy *
        lookupTxVis(x0, y1, gid, W, H, tx_topk_ids, tx_topk_vis, tx_topk_count);

    acc += fx * fy *
        lookupTxVis(x1, y1, gid, W, H, tx_topk_ids, tx_topk_vis, tx_topk_count);

    return acc;
}

//图像方向渲染函数
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderImgCUDA(
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
    int* __restrict__ accum_cnts
)
{
    auto block = cg::this_thread_block();
    const uint32_t horizontal_blocks = (img_width + BLOCK_X - 1) / BLOCK_X;

    const uint2 pix_min = {
        block.group_index().x * BLOCK_X,
        block.group_index().y * BLOCK_Y
    };
    const uint2 pix = {
        pix_min.x + block.thread_index().x,
        pix_min.y + block.thread_index().y
    };

    const bool inside = (pix.x < img_width && pix.y < img_height);
    bool done = !inside;

    const uint32_t pix_id = pix.y * img_width + pix.x;
    const int th_bin = (int)pix.x;
    const int r_bin  = (int)pix.y;

    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int toDo = range.y - range.x;

    __shared__ uint32_t sh_id[BLOCK_SIZE];
    __shared__ float    sh_intensity[BLOCK_SIZE];
    __shared__ float    sh_img_params[BLOCK_SIZE][SONAR_IMG_PARAM_STRIDE];
    __shared__ float3   sh_Tu[BLOCK_SIZE];
    __shared__ float3   sh_Tv[BLOCK_SIZE];
    __shared__ float3   sh_Tw[BLOCK_SIZE];

#if IMG_ACCUM_MODE == 0
    float accum_sum = 0.0f;
    int accum_cnt = 0;
#endif
#if IMG_ACCUM_MODE == 1
    float accum_sum = 0.0f;
#endif
#if IMG_ACCUM_MODE == 2
    float accum_lse = 1.0f;
    float accum_lse_max = 0.0f;
#endif


    for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
    {
        if (__syncthreads_count(done) == BLOCK_SIZE) break;

        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const uint32_t gid = point_list[range.x + progress];
            sh_id[block.thread_rank()] = gid;
            sh_intensity[block.thread_rank()] = intensity[gid];

            #pragma unroll
            for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                sh_img_params[block.thread_rank()][k] = img_params[gid * SONAR_IMG_PARAM_STRIDE + k];

            sh_Tu[block.thread_rank()] = {
                tx_transMats[9 * gid + 0],
                tx_transMats[9 * gid + 1],
                tx_transMats[9 * gid + 2]
            };
            sh_Tv[block.thread_rank()] = {
                tx_transMats[9 * gid + 3],
                tx_transMats[9 * gid + 4],
                tx_transMats[9 * gid + 5]
            };
            sh_Tw[block.thread_rank()] = {
                tx_transMats[9 * gid + 6],
                tx_transMats[9 * gid + 7],
                tx_transMats[9 * gid + 8]
            };
        }
        block.sync();

        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
        {
            const uint32_t gid = sh_id[j];

            float2 uv_sol[2];
            const int nsol = imgPixToGs(
                th_bin, r_bin,
                sh_img_params[j],
                GS_CUTOFF,
                uv_sol);

            if (nsol <= 0) continue;

            const float T9[9] = {//？？这里为什么不用glm库，用glm库可以统一调用project_gs2tx，删除多余函数
                sh_Tu[j].x, sh_Tu[j].y, sh_Tu[j].z,
                sh_Tv[j].x, sh_Tv[j].y, sh_Tv[j].z,
                sh_Tw[j].x, sh_Tw[j].y, sh_Tw[j].z
            };

				#pragma unroll
				for (int s = 0; s < nsol; ++s)
			{
                float2 tx_pix;
                if (!project_Gs2Tx(T9, uv_sol[s].x, uv_sol[s].y, tx_pix))
                    continue;

                const float vis_weight = lookupTxVisBilinear(
                    tx_pix.x, tx_pix.y, gid,
                    tx_width, tx_height,
                    tx_topk_ids, tx_topk_vis_weight, tx_topk_count);

				if (vis_weight <= 0.0f) continue;

				const float sol_contrib = vis_weight * sh_intensity[j];
				if (sol_contrib <= 0.0f) continue;

			#if IMG_ACCUM_MODE == 0   // mean
				accum_sum += sol_contrib;
				accum_cnt += 1;
			#elif IMG_ACCUM_MODE == 1 // sum
				accum_sum += sol_contrib;
			#elif IMG_ACCUM_MODE == 2 // log(1 + sum(exp(sol_contrib)))
                if (sol_contrib <= accum_lse_max)
                {
                    accum_lse += expf(sol_contrib - accum_lse_max);
                }
                else
                {
                    accum_lse = accum_lse * expf(accum_lse_max - sol_contrib) + 1.0f;
                    accum_lse_max = sol_contrib;
                }
			#endif
			}
        }
    }

    if (inside)
    {
        float out=0.0f;
    #if IMG_ACCUM_MODE == 0
        if (accum_cnt > 0) out = accum_sum / (float)accum_cnt;
        accum_cnts[pix_id] = accum_cnt;
    #elif IMG_ACCUM_MODE == 1
        out = accum_sum;
    #elif IMG_ACCUM_MODE == 2
        out = accum_lse_max + logf(accum_lse);
    #endif
        out_intensity[pix_id] = out;
    }
}


void FORWARD::preprocess(
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
)
{
    preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P,
        gs_trans,
        gs_quats,
	        gs_scales,
	        gs_scale_modifier,
	        lowpass_variance,
	        gs_opacities,
        gs_reflections,
        viewmatrix,
        projmatrix,
        cam_pos,
        tx_width, tx_height,
        tan_fovx, tan_fovy,
        theta_min,
        inv_dtheta,
        range_min,
        inv_dr,
        range_max,
        tx_grid,
        img_grid,
        gs_intensities,
        gs_normals,
        r_raddi,
	        tx_bounds,
	        tx_depths,
	        tx_centers_2d,
	        tx_centers,
        tx_axis_u,
        tx_axis_v,
        tx_transMats,
        tx_tiles_touched,
        img_bounds,
        img_params,
        img_tiles_touched
    );
}
void FORWARD::render_tx(
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
	float* out_others)
{
    renderTxCUDA<NUM_CHANNELS><<<grid, block>>>(
    ranges,
    point_list,
    W, H,
    range_max,
	    cam_pos,
	    tx_transMats,
	    tx_centers_2d,
	    tx_centers,
    tx_axis_u,
    tx_axis_v,
	    gs_normals,
	    opacity,
	    lowpass_variance,
	    topk_ids,
    topk_vis_weight,
    topk_count,
    out_others);
}
void FORWARD::render_img(
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
    int* __restrict__ accum_cnt
	)
{
    renderImgCUDA<NUM_CHANNELS><<<grid, block>>>(
    ranges,
    point_list,
    tx_width, tx_height,
    img_width, img_height,
    img_params,     // P * 11
    tx_transMats,   // P * 9
    intensity,      // P
    normal,        // P，可先不用，但保留独立接口
    tx_topk_ids,
    tx_topk_vis_weight,
    tx_topk_count,
    out_intensity,
	accum_cnt
);
}
