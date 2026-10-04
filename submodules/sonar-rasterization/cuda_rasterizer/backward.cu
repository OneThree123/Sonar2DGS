#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <glm/glm.hpp>
namespace cg = cooperative_groups;

__device__ __constant__ float2 b_theta_dir[MAX_AZIMUTH_BINS];
__device__ __constant__ float  b_range_val[MAX_RANGE_BINS];

// 将三角函数常量从主机复制到设备常量内存
void copyConstants2Backward(float2* d_theta_dir, float* d_range_val, float num_theta, float num_range)
{
    cudaMemcpyToSymbol(b_theta_dir, d_theta_dir, num_theta * sizeof(float2));
    cudaMemcpyToSymbol(b_range_val, d_range_val, num_range * sizeof(float));
}

__device__ __forceinline__ glm::vec3 rotate_world_to_sonar_bwd(
    const glm::vec3& v_w,
    const float* viewmatrix)
{
    return glm::vec3(
        viewmatrix[0] * v_w.x + viewmatrix[4] * v_w.y + viewmatrix[8]  * v_w.z,
        viewmatrix[1] * v_w.x + viewmatrix[5] * v_w.y + viewmatrix[9]  * v_w.z,
        viewmatrix[2] * v_w.x + viewmatrix[6] * v_w.y + viewmatrix[10] * v_w.z
    );
}

__device__ __forceinline__ glm::vec3 rotate_sonar_to_world_bwd(
    const glm::vec3& v_s,
    const float* viewmatrix)
{
    return glm::vec3(
        viewmatrix[0] * v_s.x + viewmatrix[1] * v_s.y + viewmatrix[2]  * v_s.z,
        viewmatrix[4] * v_s.x + viewmatrix[5] * v_s.y + viewmatrix[6]  * v_s.z,
        viewmatrix[8] * v_s.x + viewmatrix[9] * v_s.y + viewmatrix[10] * v_s.z
    );
}

__device__ __forceinline__ bool txPixToGs(
    int px, int py,
    const float* T9,
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

__device__ __forceinline__ void txPixToGs_vjp(
    int px, int py,
    const float* T9,
    const float2& uv,
    const float2& dL_duv,
    float* dL_dT9)
{
    float3 Tu = {T9[0], T9[1], T9[2]};
    float3 Tv = {T9[3], T9[4], T9[5]};
    float3 Tw = {T9[6], T9[7], T9[8]};

    const float3 k = px * Tw - Tu;
    const float3 l = py * Tw - Tv;
    const float3 p = cross(k, l);
    if (fabsf(p.z) < 1e-8f) return;

    const float dsx_pz = dL_duv.x / p.z;
    const float dsy_pz = dL_duv.y / p.z;
    const float3 dL_dp = {
        dsx_pz,
        dsy_pz,
        -(dsx_pz * uv.x + dsy_pz * uv.y)
    };
    const float3 dL_dk = cross(l, dL_dp);
    const float3 dL_dl = cross(dL_dp, k);

    const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
    const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
    const float3 dL_dTw = {
        px * dL_dk.x + py * dL_dl.x,
        px * dL_dk.y + py * dL_dl.y,
        px * dL_dk.z + py * dL_dl.z
    };

	    dL_dT9[0] += dL_dTu.x; dL_dT9[1] += dL_dTu.y; dL_dT9[2] += dL_dTu.z;
	    dL_dT9[3] += dL_dTv.x; dL_dT9[4] += dL_dTv.y; dL_dT9[5] += dL_dTv.z;
		    dL_dT9[6] += dL_dTw.x; dL_dT9[7] += dL_dTw.y; dL_dT9[8] += dL_dTw.z;
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

__device__ __forceinline__ bool compute_tx_hit_range(
    const int px,
    const int py,
    const float* T9,
    const float3& center,
    const float3& axis_u,
    const float3& axis_v,
    const glm::vec3& cam_pos,
    float2& uv,
    float& depth)
{
    if (!txPixToGs(px, py, T9, uv))
        return false;

    const float3 point = point_on_splat_world(center, axis_u, axis_v, uv);
    const float3 ray = make_float3(
        point.x - cam_pos.x,
        point.y - cam_pos.y,
        point.z - cam_pos.z);
    depth = sqrtf(fmaxf(dot3(ray, ray), 1e-12f));
    return true;
}

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

__device__ __forceinline__ void project_Gs2Tx_vjp(
    const float* T9,
    const float u,
    const float v,
    const float2& pix,
    const float2& dL_dpix,
    float2& dL_duv,
    float* dL_dT9)
{
    const float w = T9[6] * u + T9[7] * v + T9[8];
    if (fabsf(w) < 1e-8f) return;

    const float inv_w = 1.0f / w;
    const float dL_dn0 = dL_dpix.x * inv_w;
    const float dL_dn1 = dL_dpix.y * inv_w;
    const float dL_dw  = -(dL_dpix.x * pix.x + dL_dpix.y * pix.y) * inv_w;

    dL_dT9[0] += u * dL_dn0;
    dL_dT9[1] += v * dL_dn0;
    dL_dT9[2] += dL_dn0;
    dL_dT9[3] += u * dL_dn1;
    dL_dT9[4] += v * dL_dn1;
    dL_dT9[5] += dL_dn1;
    dL_dT9[6] += u * dL_dw;
    dL_dT9[7] += v * dL_dw;
    dL_dT9[8] += dL_dw;

    dL_duv.x += T9[0] * dL_dn0 + T9[3] * dL_dn1 + T9[6] * dL_dw;
    dL_duv.y += T9[1] * dL_dn0 + T9[4] * dL_dn1 + T9[7] * dL_dw;
}

__device__ __forceinline__ int imgPixToGs(
    const int th_bin,
    const int r_bin,
    const float* params,
    const float cutoff,
    float2 out_uv[2],
    int out_sign[2])
{
    const float c = b_theta_dir[th_bin].x;
    const float s = b_theta_dir[th_bin].y;
    const float r = b_range_val[r_bin];

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
    if (rho2 < 0.0f) return 0;

    const float d2 = alpha * alpha + beta * beta;
    if (d2 < 1e-10f) return 0;

    const float q = alpha * ou + beta * ov - gamma;
    float delta = d2 * rho2 - q * q;
    if (delta < 0.0f)
    {
        if (delta > -1e-6f) delta = 0.0f;
        else return 0;
    }

    const float root = sqrtf(delta);
    const float inv_d2 = 1.0f / d2;

    int nsol = 0;
    {
        const float u_phys = q * alpha * inv_d2 - ou - beta  * root * inv_d2;
        const float v_phys = q * beta  * inv_d2 - ov + alpha * root * inv_d2;
        const float u_std = u_phys * inv_su;
        const float v_std = v_phys * inv_sv;
        if (u_std * u_std + v_std * v_std <= cutoff * cutoff)
        {
            out_uv[nsol] = make_float2(u_std, v_std);
            out_sign[nsol] = -1;
            nsol++;
        }

    }
    if (root > 1e-7f)
    {
        const float u_phys = q * alpha * inv_d2 - ou + beta  * root * inv_d2;
        const float v_phys = q * beta  * inv_d2 - ov - alpha * root * inv_d2;
        const float u_std = u_phys * inv_su;
        const float v_std = v_phys * inv_sv;
        if (u_std * u_std + v_std * v_std <= cutoff * cutoff)
        {
            out_uv[nsol] = make_float2(u_std, v_std);
            out_sign[nsol] = 1;
            nsol++;
        }
    }
    return nsol;
}

__device__ __forceinline__ void imgPixToGs_solution_vjp(
    const int th_bin,
    const int r_bin,
    const float* params,
    const int sign_root,
    const float2& uv_sol,
    const float2& dL_duv_sol,
    float* dL_dparams)
{
    const float c = b_theta_dir[th_bin].x;
    const float s = b_theta_dir[th_bin].y;
    const float r = b_range_val[r_bin];

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
    const float d2 = alpha * alpha + beta * beta;
    if (rho2 < 0.0f || d2 < 1e-10f) return;

    const float q = alpha * ou + beta * ov - gamma;
    float delta = d2 * rho2 - q * q;
    if (delta < 0.0f)
    {
        if (delta > -1e-6f) delta = 0.0f;
        else return;
    }
    const float root = sqrtf(delta);
    const float inv_d2 = 1.0f / d2;

    const float u_phys = uv_sol.x / fmaxf(inv_su, 1e-12f);
    const float v_phys = uv_sol.y / fmaxf(inv_sv, 1e-12f);

    float dL_du_phys = dL_duv_sol.x * inv_su;
    float dL_dv_phys = dL_duv_sol.y * inv_sv;
    dL_dparams[IMG_INV_SU] += dL_duv_sol.x * u_phys;
    dL_dparams[IMG_INV_SV] += dL_duv_sol.y * v_phys;

    const float t1 = q * alpha * inv_d2;
    const float t2 = q * beta  * inv_d2;
    const float t3 = root * inv_d2;

    float dL_dt1 = dL_du_phys;
    float dL_dt2 = dL_dv_phys;
    float dL_dt3 = sign_root * beta * dL_du_phys - sign_root * alpha * dL_dv_phys;
    float dL_dalpha = -sign_root * t3 * dL_dv_phys;
    float dL_dbeta  =  sign_root * t3 * dL_du_phys;
    float dL_dou    = -dL_du_phys;
    float dL_dov    = -dL_dv_phys;

    float dL_dq = alpha * inv_d2 * dL_dt1 + beta * inv_d2 * dL_dt2;
    dL_dalpha += q * inv_d2 * dL_dt1;
    dL_dbeta  += q * inv_d2 * dL_dt2;
    float dL_dinv_d2 = q * alpha * dL_dt1 + q * beta * dL_dt2;

    float dL_droot = inv_d2 * dL_dt3;
    dL_dinv_d2 += root * dL_dt3;

    float dL_dd2 = -dL_dinv_d2 * inv_d2 * inv_d2;
    float dL_ddelta = 0.0f;
    if (root > 1e-8f)
        dL_ddelta = 0.5f * dL_droot / root;

    dL_dd2 += rho2 * dL_ddelta;
    float dL_drho2 = d2 * dL_ddelta;
    dL_dq += -2.0f * q * dL_ddelta;

    dL_dalpha += ou * dL_dq;
    dL_dou    += alpha * dL_dq;
    dL_dbeta  += ov * dL_dq;
    dL_dov    += beta * dL_dq;
    float dL_dgamma = -dL_dq;

    dL_dalpha += 2.0f * alpha * dL_dd2;
    dL_dbeta  += 2.0f * beta  * dL_dd2;
    float dL_dh = -2.0f * h * dL_drho2;

    dL_dparams[IMG_RUX] += -s * dL_dalpha;
    dL_dparams[IMG_RUY] +=  c * dL_dalpha;
    dL_dparams[IMG_RVX] += -s * dL_dbeta;
    dL_dparams[IMG_RVY] +=  c * dL_dbeta;
    dL_dparams[IMG_OX]  += -s * dL_dgamma;
    dL_dparams[IMG_OY]  +=  c * dL_dgamma;
    dL_dparams[IMG_OU]  += dL_dou;
    dL_dparams[IMG_OV]  += dL_dov;
    dL_dparams[IMG_H]   += dL_dh;
}

__device__ __forceinline__ float lookupTxVis(
    const int tx_x,
    const int tx_y,
    const uint32_t gid,
    const int W,
    const int H,
    const uint32_t* tx_topk_ids,
    const float* tx_topk_vis,
    const uint8_t* tx_topk_count,
    int& out_index)
{
    if (tx_x < 0 || tx_x >= W || tx_y < 0 || tx_y >= H) {
        out_index = -1;
        return 0.0f;
    }

    const int pix_id = tx_y * W + tx_x;
    const int cnt = (int)tx_topk_count[pix_id];
    const int base = pix_id * TX_TOPK;

    for (int k = 0; k < cnt; ++k)
    {
        if (tx_topk_ids[base + k] == gid) {
            out_index = base + k;
            return tx_topk_vis[out_index];
        }
    }

    out_index = -1;
    return 0.0f;
}
__device__ __forceinline__ float warpReduceSum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

template<int N>
__device__ __forceinline__ void warpReduceSumArray(float (&v)[N])
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        #pragma unroll
        for (int k = 0; k < N; ++k)
            v[k] += __shfl_down_sync(0xffffffff, v[k], offset);
    }
}
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderImgBackwardCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    const int tx_width, const int tx_height,
    const int img_width, const int img_height,
    const float* __restrict__ img_params,
    const float* __restrict__ tx_transMats,
    const float* __restrict__ intensity,
    const uint32_t* __restrict__ tx_topk_ids,
    const float* __restrict__ tx_topk_vis_weight,
    const uint8_t* __restrict__ tx_topk_count,
    const float* __restrict__ dL_dout_intensity,
    const int* __restrict__ accum_cnts,
    const float* __restrict__ out_intensity,
    float* __restrict__ gs_dL_dimg_params,
    float* __restrict__ gs_dL_dtx_transMat,
    float* __restrict__ gs_dL_dintensity,
    float* __restrict__ dL_dtx_topk_vis_weight)
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
    float dL_dout = 0.0f;
    float coef = 0.0f;
    float out = 0.0f;

    const int lane = block.thread_rank() & 31;
    const int wid  = block.thread_rank() >> 5;

    __shared__ float sh_warp_dintensity[NUM_WARPS];
    __shared__ float sh_warp_dT9[NUM_WARPS][9];
    __shared__ float sh_warp_dparams[NUM_WARPS][SONAR_IMG_PARAM_STRIDE];

    const uint32_t pix_id = pix.y * img_width + pix.x;
    const int th_bin = (int)pix.x;
    const int r_bin  = (int)pix.y;
    const bool inside = (pix.x < img_width && pix.y < img_height);

    if (inside)
    {
        dL_dout = dL_dout_intensity[pix_id];
    #if IMG_ACCUM_MODE == 0
        const int accum_cnt = accum_cnts[pix_id];
        coef = (accum_cnt > 0) ? (1.0f / float(accum_cnt)) : 0.0f;
    #elif IMG_ACCUM_MODE == 2
        out = out_intensity[pix_id];
    #endif
    }


    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int toDo = range.y - range.x;

    __shared__ uint32_t sh_id[BLOCK_SIZE];
    __shared__ float    sh_intensity[BLOCK_SIZE];
    __shared__ float    sh_img_params[BLOCK_SIZE][SONAR_IMG_PARAM_STRIDE];
    __shared__ float3   sh_Tu[BLOCK_SIZE];
    __shared__ float3   sh_Tv[BLOCK_SIZE];
    __shared__ float3   sh_Tw[BLOCK_SIZE];

    for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
    {
        //读取所有高斯球数据到shared memory
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const uint32_t gid = point_list[range.x + progress];
            sh_id[block.thread_rank()] = gid;
            sh_intensity[block.thread_rank()] = intensity[gid];
            #pragma unroll
            for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                sh_img_params[block.thread_rank()][k] = img_params[gid * SONAR_IMG_PARAM_STRIDE + k];
            sh_Tu[block.thread_rank()] = {tx_transMats[9 * gid + 0], tx_transMats[9 * gid + 1], tx_transMats[9 * gid + 2]};
            sh_Tv[block.thread_rank()] = {tx_transMats[9 * gid + 3], tx_transMats[9 * gid + 4], tx_transMats[9 * gid + 5]};
            sh_Tw[block.thread_rank()] = {tx_transMats[9 * gid + 6], tx_transMats[9 * gid + 7], tx_transMats[9 * gid + 8]};
        }
        block.sync();
        //逐个高斯串行计算
        for (int j = 0; j < min(BLOCK_SIZE, toDo); ++j)
        {
            const uint32_t gid = sh_id[j];

            // ---------- 每个线程对“当前 gid”先局部累加 ----------
            float local_dintensity = 0.0f;
            float local_dT9[9] = {0.0f};
            float local_dparams[SONAR_IMG_PARAM_STRIDE] = {0.0f};

            if (inside)
            {
                float2 uv_sol[2];
                int sign_sol[2];
                const int nsol = imgPixToGs(th_bin, r_bin, sh_img_params[j], GS_CUTOFF, uv_sol, sign_sol);
                if (nsol > 0)
                {
                    const float T9[9] = {
                        sh_Tu[j].x, sh_Tu[j].y, sh_Tu[j].z,
                        sh_Tv[j].x, sh_Tv[j].y, sh_Tv[j].z,
                        sh_Tw[j].x, sh_Tw[j].y, sh_Tw[j].z
                    };

                    for (int sidx = 0; sidx < nsol; ++sidx)
                    {
                        float2 tx_pix;
                        if (!project_Gs2Tx(T9, uv_sol[sidx].x, uv_sol[sidx].y, tx_pix))
                            continue;

                        const int x0 = (int)floorf(tx_pix.x);
                        const int y0 = (int)floorf(tx_pix.y);
                        const int x1 = x0 + 1;
                        const int y1 = y0 + 1;
                        const float fx = tx_pix.x - x0;
                        const float fy = tx_pix.y - y0;

                        int idx00, idx10, idx01, idx11;
                        const float v00 = lookupTxVis(x0, y0, gid, tx_width, tx_height, tx_topk_ids, tx_topk_vis_weight, tx_topk_count, idx00);
                        const float v10 = lookupTxVis(x1, y0, gid, tx_width, tx_height, tx_topk_ids, tx_topk_vis_weight, tx_topk_count, idx10);
                        const float v01 = lookupTxVis(x0, y1, gid, tx_width, tx_height, tx_topk_ids, tx_topk_vis_weight, tx_topk_count, idx01);
                        const float v11 = lookupTxVis(x1, y1, gid, tx_width, tx_height, tx_topk_ids, tx_topk_vis_weight, tx_topk_count, idx11);

                        const float vis = (1.0f - fx) * (1.0f - fy) * v00 +
                                        fx * (1.0f - fy) * v10 +
                                        (1.0f - fx) * fy * v01 +
                                        fx * fy * v11;

                        const float sol_contrib = vis * sh_intensity[j];
                        if (vis <= 0.0f || sol_contrib <= 0.0f) continue;

                    #if IMG_ACCUM_MODE == 0
                        const float grad_sol = dL_dout * coef;
                    #elif IMG_ACCUM_MODE == 1
                        const float grad_sol = dL_dout;
                    #elif IMG_ACCUM_MODE == 2
                        // out = log(1 + sum(exp(sol_contrib))), so d out / d sol = exp(sol - out).
                        const float grad_sol = dL_dout * expf(sol_contrib - out);
                    #endif

                        // 1) intensity 先线程内累加
                        local_dintensity += vis * grad_sol;

                        const float dL_dvis = sh_intensity[j] * grad_sol;

                        // 2) 这个仍然保留全局 atomicAdd
                        // 因为索引是 idx00/10/01/11，不是单纯 gid
                        if (idx00 >= 0) atomicAdd(&dL_dtx_topk_vis_weight[idx00], (1.0f - fx) * (1.0f - fy) * dL_dvis);
                        if (idx10 >= 0) atomicAdd(&dL_dtx_topk_vis_weight[idx10], fx * (1.0f - fy) * dL_dvis);
                        if (idx01 >= 0) atomicAdd(&dL_dtx_topk_vis_weight[idx01], (1.0f - fx) * fy * dL_dvis);
                        if (idx11 >= 0) atomicAdd(&dL_dtx_topk_vis_weight[idx11], fx * fy * dL_dvis);

                        const float dvis_dx = (1.0f - fy) * (v10 - v00) + fy * (v11 - v01);
                        const float dvis_dy = (1.0f - fx) * (v01 - v00) + fx * (v11 - v10);
                        const float2 dL_dtx_pix = {dL_dvis * dvis_dx, dL_dvis * dvis_dy};

                        float2 dL_duv = {0.0f, 0.0f};
                        float dL_dT9_local[9] = {0.0f};
                        project_Gs2Tx_vjp(
                            T9, uv_sol[sidx].x, uv_sol[sidx].y,
                            tx_pix, dL_dtx_pix, dL_duv, dL_dT9_local);

                        float dL_dparams_local[SONAR_IMG_PARAM_STRIDE] = {0.0f};
                        const int sign_root = (sidx == 0) ? -1 : +1;
                        imgPixToGs_solution_vjp(
                            th_bin, r_bin, sh_img_params[j], sign_sol[sidx],
                            uv_sol[sidx], dL_duv, dL_dparams_local);

                        #pragma unroll
                        for (int k = 0; k < 9; ++k)
                            local_dT9[k] += dL_dT9_local[k];

                        #pragma unroll
                        for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                            local_dparams[k] += dL_dparams_local[k];
                    }
                }
            }

            // ---------- 先做 warp 内规约 ----------
            local_dintensity = warpReduceSum(local_dintensity);
            warpReduceSumArray<9>(local_dT9);
            warpReduceSumArray<SONAR_IMG_PARAM_STRIDE>(local_dparams);

            // 每个 warp 的 lane0 写到 shared memory
            if (lane == 0)
            {
                sh_warp_dintensity[wid] = local_dintensity;

                #pragma unroll
                for (int k = 0; k < 9; ++k)
                    sh_warp_dT9[wid][k] = local_dT9[k];

                #pragma unroll
                for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                    sh_warp_dparams[wid][k] = local_dparams[k];
            }
            block.sync();

            // ---------- 再由 warp0 做 block 级规约 ----------
            if (wid == 0)
            {
                float block_dintensity = (lane < NUM_WARPS) ? sh_warp_dintensity[lane] : 0.0f;
                float block_dT9[9] = {0.0f};
                float block_dparams[SONAR_IMG_PARAM_STRIDE] = {0.0f};

                if (lane < NUM_WARPS)
                {
                    #pragma unroll
                    for (int k = 0; k < 9; ++k)
                        block_dT9[k] = sh_warp_dT9[lane][k];

                    #pragma unroll
                    for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                        block_dparams[k] = sh_warp_dparams[lane][k];
                }

                block_dintensity = warpReduceSum(block_dintensity);
                warpReduceSumArray<9>(block_dT9);
                warpReduceSumArray<SONAR_IMG_PARAM_STRIDE>(block_dparams);

                if (lane == 0)
                {
                    atomicAdd(&gs_dL_dintensity[gid], block_dintensity);

                    #pragma unroll
                    for (int k = 0; k < 9; ++k)
                        atomicAdd(&gs_dL_dtx_transMat[gid * 9 + k], block_dT9[k]);

                    #pragma unroll
                    for (int k = 0; k < SONAR_IMG_PARAM_STRIDE; ++k)
                        atomicAdd(&gs_dL_dimg_params[gid * SONAR_IMG_PARAM_STRIDE + k], block_dparams[k]);
                }
            }
            block.sync();
        }
    }
}
constexpr uint32_t EMPTY_GID = 0xFFFFFFFFu;
constexpr int TX_BLOCK_HASH_CAP = 512;   // 先从 512 开始测

__device__ __forceinline__ int findOrInsertBlockGid(
    uint32_t gid,
    uint32_t* sh_gid)
{
    int slot = (int)((gid * 2654435761u) & (TX_BLOCK_HASH_CAP - 1));

    #pragma unroll 1
    for (int it = 0; it < TX_BLOCK_HASH_CAP; ++it)
    {
        uint32_t cur = sh_gid[slot];
        if (cur == gid) return slot;
        if (cur == EMPTY_GID) {
            uint32_t old = atomicCAS(&sh_gid[slot], EMPTY_GID, gid);
            if (old == EMPTY_GID || old == gid) return slot;
        }
        slot = (slot + 1) & (TX_BLOCK_HASH_CAP - 1);
    }
    return -1; // hash 满了，走 fallback
}
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderTxBackwardCUDA(
    int W, int H,
	    float range_max,
		    const glm::vec3* __restrict__ cam_pos,
		    const float* __restrict__ tx_transMats,
		    const float2* __restrict__ tx_centers_2d,
		    const float* __restrict__ tx_depths,
	    const float3* __restrict__ tx_centers,
	    const float3* __restrict__ tx_axis_u,
	    const float3* __restrict__ tx_axis_v,
		    const float3* __restrict__ gs_normals,
	    const float* __restrict__ opacity,
	    const float lowpass_variance,
	    const uint32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_vis_weight,
    const uint8_t* __restrict__ topk_count,
    const float* __restrict__ dL_dothers,
	    const float* __restrict__ dL_dtx_vis_weight,
	    float* __restrict__ dL_dtx_trans,
	    float* __restrict__ dL_dgs_dep,
	    float* __restrict__ dL_dtx_center,
	    float* __restrict__ dL_dtx_axis_u,
	    float* __restrict__ dL_dtx_axis_v,
	    float* __restrict__ dL_dgs_nor,
    float* __restrict__ dL_dgs_opa)
{
    //计算像素值
    auto block = cg::this_thread_block();
    const uint2 pix = {
        block.group_index().x * BLOCK_X + block.thread_index().x,
        block.group_index().y * BLOCK_Y + block.thread_index().y
    };    

	    //定义共享内存与初始化
	    __shared__ uint32_t sh_gid[TX_BLOCK_HASH_CAP];
	    __shared__ float sh_dep[TX_BLOCK_HASH_CAP];
	    __shared__ float sh_nor[TX_BLOCK_HASH_CAP][3];
    __shared__ float sh_opa[TX_BLOCK_HASH_CAP];
    __shared__ float sh_T9[TX_BLOCK_HASH_CAP][9];
    const int tid = block.thread_rank();

    for (int i = tid; i < TX_BLOCK_HASH_CAP; i += BLOCK_X * BLOCK_Y)
	    {
	        sh_gid[i] = EMPTY_GID;
	        sh_dep[i] = 0.0f;
	        sh_opa[i] = 0.0f;

        #pragma unroll
        for (int c = 0; c < 3; ++c)
            sh_nor[i][c] = 0.0f;

        #pragma unroll
        for (int t = 0; t < 9; ++t)
            sh_T9[i][t] = 0.0f;
    }
    block.sync();
    //其他变量初始化
    const bool inside = (pix.x < W && pix.y < H);

    const uint32_t pix_id = W * pix.y + pix.x;
    const int base   = pix_id * TX_TOPK;
    const int HW     = W * H;
    const float eps  = 1e-8f;


	    float T_before[TX_TOPK];
	    float hit_depth[TX_TOPK];
	    float2 hit_uv[TX_TOPK];
	    bool hit_ok[TX_TOPK];
    const float inv_range_max = 1.f / fmaxf(range_max, eps);
    float prefix_w = 0.f;
    float prefix_wm = 0.f;
    float prefix_wm2 = 0.f;
    if (inside){
        const int K = (int)topk_count[pix_id];
        // ---------- 小正向循环：只遍历已保存的 topk ----------
        for (int k = 0; k < K; ++k)
        {
            const uint32_t gid = topk_ids[base + k];
            const float wk = topk_vis_weight[base + k];

            T_before[k] = fmaxf(1.f - prefix_w, eps);
            prefix_w += wk;

	            const float* T9 = tx_transMats + 9 * gid;
	            hit_ok[k] = compute_tx_hit_range(
	                pix.x, pix.y, T9,
	                tx_centers[gid],
	                tx_axis_u[gid],
	                tx_axis_v[gid],
	                *cam_pos,
	                hit_uv[k],
	                hit_depth[k]);
	            const float depth = hit_ok[k] ? hit_depth[k] : tx_depths[gid];
	            const float z = depth * inv_range_max;
            const float mk = fminf(fmaxf(z, 0.f), 1.f);

            prefix_wm += mk * wk;
            prefix_wm2 += mk * mk * wk;
        }

        // ---------- 读像素导数 ----------
        const float g_depth = dL_dothers[pix_id + DEPTH_OFFSET * HW];
        const float g_alpha = dL_dothers[pix_id + ALPHA_OFFSET * HW];
        const float g_dist  = dL_dothers[pix_id + DISTORTION_OFFSET * HW];
        const float3 g_norm = {
            dL_dothers[pix_id + (NORMAL_OFFSET + 0) * HW],
            dL_dothers[pix_id + (NORMAL_OFFSET + 1) * HW],
            dL_dothers[pix_id + (NORMAL_OFFSET + 2) * HW]
        };

        // ---------- 逆序传播 ----------
        float aT = 0.f;
        float suffix_w = 0.f;
        float suffix_wm = 0.f;
        float suffix_wm2 = 0.f;

        for (int k = K - 1; k >= 0; --k)
        {
            const uint32_t gid = topk_ids[base + k];
            const float wk = topk_vis_weight[base + k];
            const float Tprev = T_before[k];

	            const float depth = hit_ok[k] ? hit_depth[k] : tx_depths[gid];
	            const float3 normal = gs_normals[gid];

            const float z = depth * inv_range_max;
            const float mk = fminf(fmaxf(z, 0.f), 1.f);
            const float dmk_dd = (z > 0.f && z < 1.f) ? inv_range_max : 0.f;

            float alpha_k = wk / Tprev;
            alpha_k = fminf(alpha_k, 0.99f);   // 建议加上，和 forward 保持一致

            const float prefix_w_before = prefix_w - suffix_w - wk;
            const float prefix_wm_before = prefix_wm - suffix_wm - wk * mk;
            const float prefix_wm2_before =
                prefix_wm2 - suffix_wm2 - wk * mk * mk;
            const float other_w = prefix_w_before + suffix_w;
            const float other_wm = prefix_wm_before + suffix_wm;
            const float other_wm2 = prefix_wm2_before + suffix_wm2;

            // Derivatives of sum_{i<j} w_i w_j (m_i - m_j)^2.
            const float d_dist_dwk =
                mk * mk * other_w + other_wm2 - 2.f * mk * other_wm;
            const float d_dist_dmk =
                2.f * wk * (mk * other_w - other_wm);

            float beta = 0.f;
            beta += dL_dtx_vis_weight[base + k];
#if !DETACH_WEIGHT
            beta += depth * g_depth;
            beta += normal.x * g_norm.x + normal.y * g_norm.y + normal.z * g_norm.z;
            beta += g_alpha;
            beta += g_dist * d_dist_dwk;
#endif

            const float d_alpha = Tprev * (beta - aT);
            aT = alpha_k * beta + (1.f - alpha_k) * aT;

            const float ddep =
                wk * g_depth +
                g_dist * d_dist_dmk * dmk_dd;

            const float dnor0 = wk * g_norm.x;
            const float dnor1 = wk * g_norm.y;
            const float dnor2 = wk * g_norm.z;

	            float2 dL_duv = {0.0f, 0.0f};
	            if (hit_ok[k])
	            {
	                const float2 uv_k = hit_uv[k];
	                const float3 center = tx_centers[gid];
	                const float3 axis_u = tx_axis_u[gid];
	                const float3 axis_v = tx_axis_v[gid];
	                const float3 point = point_on_splat_world(center, axis_u, axis_v, uv_k);
	                const float inv_depth = 1.0f / fmaxf(depth, eps);
	                const float3 unit_ray = make_float3(
	                    (point.x - cam_pos->x) * inv_depth,
	                    (point.y - cam_pos->y) * inv_depth,
	                    (point.z - cam_pos->z) * inv_depth);
	                const float3 dpoint = make_float3(
	                    ddep * unit_ray.x,
	                    ddep * unit_ray.y,
	                    ddep * unit_ray.z);

	                atomicAdd(&dL_dtx_center[gid * 3 + 0], dpoint.x);
	                atomicAdd(&dL_dtx_center[gid * 3 + 1], dpoint.y);
	                atomicAdd(&dL_dtx_center[gid * 3 + 2], dpoint.z);

	                atomicAdd(&dL_dtx_axis_u[gid * 3 + 0], uv_k.x * dpoint.x);
	                atomicAdd(&dL_dtx_axis_u[gid * 3 + 1], uv_k.x * dpoint.y);
	                atomicAdd(&dL_dtx_axis_u[gid * 3 + 2], uv_k.x * dpoint.z);

	                atomicAdd(&dL_dtx_axis_v[gid * 3 + 0], uv_k.y * dpoint.x);
	                atomicAdd(&dL_dtx_axis_v[gid * 3 + 1], uv_k.y * dpoint.y);
	                atomicAdd(&dL_dtx_axis_v[gid * 3 + 2], uv_k.y * dpoint.z);

	                dL_duv.x += dot3(dpoint, axis_u);
	                dL_duv.y += dot3(dpoint, axis_v);
	            }

			            float dopa = 0.0f;
			            float dT9[9] = {0.0f};
		            const float* T9 = tx_transMats + 9 * gid;
		            bool lowpass_active = false;
		            if (hit_ok[k])
		            {
		                const float2 uv_k = hit_uv[k];
		                const float2 center_2d = tx_centers_2d[gid];
		                const float dx = (float)pix.x - center_2d.x;
		                const float dy = (float)pix.y - center_2d.y;
		                const float intrinsic_rho = uv_k.x * uv_k.x + uv_k.y * uv_k.y;
		                const float lowpass_rho =
		                    (dx * dx + dy * dy) / fmaxf(lowpass_variance, eps);
		                lowpass_active = lowpass_rho < intrinsic_rho;
		            }

		            if (alpha_k < 0.99f)
		            {
		                const float opa = fmaxf(opacity[gid], eps);
		                const float G   = alpha_k / opa;

		                dopa = G * d_alpha;

			                if (hit_ok[k])
			                {
			                    const float2 uv_k = hit_uv[k];
			                    const float barG = opa * d_alpha;
			                    if (lowpass_active)
			                    {
			                        const float2 center_2d = tx_centers_2d[gid];
			                        const float inv_lowpass_variance =
			                            1.0f / fmaxf(lowpass_variance, eps);
			                        const float2 dL_dcenter_2d = {
			                            barG * G * ((float)pix.x - center_2d.x)
			                                * inv_lowpass_variance,
			                            barG * G * ((float)pix.y - center_2d.y)
			                                * inv_lowpass_variance
			                        };
			                        float2 unused_dL_duv = {0.0f, 0.0f};
			                        project_Gs2Tx_vjp(
			                            T9,
			                            0.0f,
			                            0.0f,
			                            center_2d,
			                            dL_dcenter_2d,
			                            unused_dL_duv,
			                            dT9);
			                    }
			                    else
			                    {
			                        dL_duv.x += -barG * G * uv_k.x;
			                        dL_duv.y += -barG * G * uv_k.y;
			                    }
			                }
		            }

		            if (hit_ok[k] && (dL_duv.x != 0.0f || dL_duv.y != 0.0f))
		            {
		                txPixToGs_vjp(pix.x, pix.y, T9, hit_uv[k], dL_duv, dT9);
		            }

            suffix_w += wk;
            suffix_wm += wk * mk;
            suffix_wm2 += wk * mk * mk;

            int slot = findOrInsertBlockGid(gid, sh_gid);

	            if (slot >= 0)
	            {
		                atomicAdd(&sh_nor[slot][0], dnor0);
	                atomicAdd(&sh_nor[slot][1], dnor1);
	                atomicAdd(&sh_nor[slot][2], dnor2);

                if (dopa != 0.0f)
                    atomicAdd(&sh_opa[slot], dopa);

                #pragma unroll
                for (int t = 0; t < 9; ++t)
                    if (dT9[t] != 0.0f)
                        atomicAdd(&sh_T9[slot][t], dT9[t]);
            }
	            else
	            {
		                atomicAdd(&dL_dgs_nor[gid * 3 + 0], dnor0);
	                atomicAdd(&dL_dgs_nor[gid * 3 + 1], dnor1);
	                atomicAdd(&dL_dgs_nor[gid * 3 + 2], dnor2);

                if (dopa != 0.0f)
                    atomicAdd(&dL_dgs_opa[gid], dopa);

                #pragma unroll
                for (int t = 0; t < 9; ++t)
                    if (dT9[t] != 0.0f)
                        atomicAdd(&dL_dtx_trans[gid * 9 + t], dT9[t]);
            }
        }
    }
    block.sync();

    for (int i = tid; i < TX_BLOCK_HASH_CAP; i += BLOCK_X * BLOCK_Y)
    {
	        const uint32_t gid = sh_gid[i];
	        if (gid == EMPTY_GID) continue;

	        atomicAdd(&dL_dgs_dep[gid], sh_dep[i]);

	        atomicAdd(&dL_dgs_nor[gid * 3 + 0], sh_nor[i][0]);
        atomicAdd(&dL_dgs_nor[gid * 3 + 1], sh_nor[i][1]);
        atomicAdd(&dL_dgs_nor[gid * 3 + 2], sh_nor[i][2]);

        atomicAdd(&dL_dgs_opa[gid], sh_opa[i]);

        #pragma unroll
        for (int t = 0; t < 9; ++t)
            atomicAdd(&dL_dtx_trans[gid * 9 + t], sh_T9[i][t]);
    }
}

template<int C>
__global__ void preprocessBackwardCUDA(
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
    glm::vec2* dL_dgs_screenspace_points
)
{
    const int idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    const glm::vec3 pos = gs_trans[idx];
    const glm::vec4 quat = gs_quats[idx];
    const glm::vec2 scale = gs_scales[idx];
    const float reflection = gs_reflections[idx];

    const float su = scale.x * gs_scale_modifier;
    const float sv = scale.y * gs_scale_modifier;

    glm::mat3 R = quat_to_rotmat(quat);
    glm::mat3 S = scale_to_mat(scale, gs_scale_modifier);
    glm::mat3 L = R * S;
    // glm::mat3 L = glm::transpose(R * S);


    glm::mat3x4 M = glm::mat3x4(
        glm::vec4(L[0], 0.0f),
        glm::vec4(L[1], 0.0f),
        glm::vec4(pos, 1.0f)
    );
    glm::mat4 world2ndc = glm::mat4(
        projmatrix[0],  projmatrix[4],  projmatrix[8],  projmatrix[12],
        projmatrix[1],  projmatrix[5],  projmatrix[9],  projmatrix[13],
        projmatrix[2],  projmatrix[6],  projmatrix[10], projmatrix[14],
        projmatrix[3],  projmatrix[7],  projmatrix[11], projmatrix[15]
    );
    glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4(float(tx_width) * 0.5f, 0.0f, 0.0f, float(tx_width - 1) * 0.5f),
        glm::vec4(0.0f, float(tx_height) * 0.5f, 0.0f, float(tx_height - 1) * 0.5f),
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
    );
    glm::mat3x4 Pmat = world2ndc * ndc2pix;

    glm::vec3 dpos(0.0f);
    glm::mat3 dR(0.0f);
    glm::vec2 dscale(0.0f);
	    float dreflection = 0.0f;

	    const glm::vec3 dcenter(
	        dL_dtx_center[idx * 3 + 0],
	        dL_dtx_center[idx * 3 + 1],
	        dL_dtx_center[idx * 3 + 2]);
	    const glm::vec3 daxis_u(
	        dL_dtx_axis_u[idx * 3 + 0],
	        dL_dtx_axis_u[idx * 3 + 1],
	        dL_dtx_axis_u[idx * 3 + 2]);
	    const glm::vec3 daxis_v(
	        dL_dtx_axis_v[idx * 3 + 0],
	        dL_dtx_axis_v[idx * 3 + 1],
	        dL_dtx_axis_v[idx * 3 + 2]);
	    dpos += dcenter;
	    dR[0] += daxis_u * su;
	    dR[1] += daxis_v * sv;
	    dscale.x += glm::dot(daxis_u, R[0]) * gs_scale_modifier;
	    dscale.y += glm::dot(daxis_v, R[1]) * gs_scale_modifier;

    // tx depth
    const glm::vec3 s2gs_w = pos - (*cam_pos);
    const float range = glm::length(s2gs_w);
    const glm::vec3 normal_w = R[2];
    if (range > 1e-8f)
    {
        const glm::vec3 dir_w = s2gs_w / range;
        const float cos_term = glm::dot(dir_w, normal_w);
        if (-cos_term < 0.0f)
            return;
    }

	    if (range > 1e-8f)
	        dpos += glm::vec3(s2gs_w.x, s2gs_w.y, s2gs_w.z) * (dL_dtx_depth[idx] / range);

    // tx transmat
    glm::mat3 dT(
        dL_dtx_transMat[idx * 9 + 0], dL_dtx_transMat[idx * 9 + 1], dL_dtx_transMat[idx * 9 + 2],
        dL_dtx_transMat[idx * 9 + 3], dL_dtx_transMat[idx * 9 + 4], dL_dtx_transMat[idx * 9 + 5],
        dL_dtx_transMat[idx * 9 + 6], dL_dtx_transMat[idx * 9 + 7], dL_dtx_transMat[idx * 9 + 8]
    );
    glm::mat3x4 dM = Pmat * glm::transpose(dT);
    const glm::vec3 dL0 = glm::vec3(dM[0]);
    const glm::vec3 dL1 = glm::vec3(dM[1]);
    dpos += glm::vec3(dM[2]);
    dR[0] += dL0 * su;
    dR[1] += dL1 * sv;
    dscale.x += glm::dot(dL0, R[0]) * gs_scale_modifier;
    dscale.y += glm::dot(dL1, R[1]) * gs_scale_modifier;

    // normal + intensity
    glm::vec3 dnormal_w(
        dL_dnormal3D[idx * 3 + 0],
        dL_dnormal3D[idx * 3 + 1],
        dL_dnormal3D[idx * 3 + 2]);

    if (range > 1e-8f)
    {
        const glm::vec3 dir_w = s2gs_w / range;
        const float cos_term = glm::dot(dir_w, normal_w);
        const float front_cos = fmaxf(-cos_term, 0.0f);
        const float dI = dL_dintensity[idx];

        dreflection += front_cos * dI;

        // max(-cos, 0) 在 0 处不可导，这里取 0 作为次梯度
        if (cos_term < 0.0f)
        {
            const float coeff = -reflection * dI;

            dnormal_w += coeff * dir_w;

            const glm::vec3 ddir = coeff * normal_w;
            float3 ddir3 = {ddir.x, ddir.y, ddir.z};
            float3 ds2gs = dnormvdv(float3{s2gs_w.x, s2gs_w.y, s2gs_w.z}, ddir3);
            dpos += glm::vec3(ds2gs.x, ds2gs.y, ds2gs.z);
        }
    }
    dR[2] += dnormal_w;

    // img params
    const float* gp = dL_dimg_params + idx * SONAR_IMG_PARAM_STRIDE;
    const glm::vec3 o_s  = rotate_world_to_sonar_bwd(s2gs_w, viewmatrix);
    const glm::vec3 ru_s = rotate_world_to_sonar_bwd(R[0], viewmatrix);
    const glm::vec3 rv_s = rotate_world_to_sonar_bwd(R[1], viewmatrix);
    const glm::vec3 n_s  = rotate_world_to_sonar_bwd(R[2], viewmatrix);

    glm::vec3 do_s(0.0f), dru_s(0.0f), drv_s(0.0f), dn_s(0.0f);
    do_s.x += gp[IMG_OX];
    do_s.y += gp[IMG_OY];
    dru_s.x += gp[IMG_RUX];
    dru_s.y += gp[IMG_RUY];
    drv_s.x += gp[IMG_RVX];
    drv_s.y += gp[IMG_RVY];

    const float g_ou = gp[IMG_OU];
    const float g_ov = gp[IMG_OV];
    const float g_h  = gp[IMG_H];
    dru_s += g_ou * o_s;
    do_s  += g_ou * ru_s;
    drv_s += g_ov * o_s;
    do_s  += g_ov * rv_s;
    dn_s  += g_h  * o_s;
    do_s  += g_h  * n_s;

    float dsu = 0.0f, dsv = 0.0f;
    if (su > 1e-8f) dsu += -gp[IMG_INV_SU] / (su * su);
    if (sv > 1e-8f) dsv += -gp[IMG_INV_SV] / (sv * sv);
    dscale.x += dsu * gs_scale_modifier;
    dscale.y += dsv * gs_scale_modifier;

    float dmean2D_x = 0.0f;
    float dmean2D_y = 0.0f;
    const float rho2 = o_s.x * o_s.x + o_s.y * o_s.y;
    if (rho2 > 1e-8f)
    {
        const float inv_dtheta = float(img_width) / fmaxf(fovx, 1e-8f);
        dmean2D_x = (-o_s.y * do_s.x + o_s.x * do_s.y) / inv_dtheta;
        dmean2D_x *= 0.5f * float(img_width);
    }
    if (range > 1e-8f)
    {
        const float inv_dr = float(img_height) / fmaxf(range_max, 1e-8f);
        dmean2D_y = glm::dot(do_s, o_s / range) / inv_dr;
        dmean2D_y *= 0.5f * float(img_height);
    }

    dpos += rotate_sonar_to_world_bwd(do_s, viewmatrix);
    dR[0] += rotate_sonar_to_world_bwd(dru_s, viewmatrix);
    dR[1] += rotate_sonar_to_world_bwd(drv_s, viewmatrix);
    dR[2] += rotate_sonar_to_world_bwd(dn_s, viewmatrix);

    dL_dgs_trans[idx] = dpos;
    dL_dgs_quats[idx] = quat_to_rotmat_vjp(quat, dR);
    dL_dgs_scales[idx] = dscale;
    dL_dgs_reflections[idx] = dreflection;
    dL_dgs_screenspace_points[idx] = glm::vec2(dmean2D_x, dmean2D_y);
}

void BACKWARD::render_img(
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
        float* dL_dtx_topk_vis_weight)
{
    renderImgBackwardCUDA<NUM_CHANNELS><<<grid, block>>>(
        ranges,
        point_list,
        tx_width, tx_height,
        img_width, img_height,
        img_params,
        tx_transMats,
        intensities,
        tx_topk_ids,
        tx_topk_vis_weight,
        tx_topk_count,
        dL_dout_intensity,
        accum_cnt,
        out_intensity,
        dL_dimg_params,
        dL_dtx_transMat,
        dL_dintensity,
        dL_dtx_topk_vis_weight);
}

void BACKWARD::render_tx(
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
        float* dL_dopacity)
{
    renderTxBackwardCUDA<NUM_CHANNELS><<<grid, block>>>(
	        W, H,
		        range_max,
			        cam_pos,
			        tx_transMats,
			        tx_centers_2d,
			        tx_depths,
		        tx_centers,
		        tx_axis_u,
		        tx_axis_v,
		        gs_normals,
	        opacity,
	        lowpass_variance,
	        topk_ids,
        topk_vis_weight,
        topk_count,
        dL_dothers,
        dL_dtx_topk_vis_weight,
		        dL_dtx_transMat,
		        dL_dtx_depth,
		        dL_dtx_center,
		        dL_dtx_axis_u,
		        dL_dtx_axis_v,
		        dL_dnormal3D,
        dL_dopacity);
}

void BACKWARD::preprocess(
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
    glm::vec2* dL_dgs_screenspace_points
)
{
    preprocessBackwardCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
        P,
        gs_trans,
        gs_quats,
        gs_scales,
        gs_scale_modifier,
        gs_reflections,
        viewmatrix,
        projmatrix,
        cam_pos,
        tx_width, tx_height,
        img_width, img_height,
        fovx,
        range_max,
	        dL_dnormal3D,
		        dL_dintensity,
		        dL_dtx_transMat,
		        dL_dtx_depth,
		        dL_dtx_center,
		        dL_dtx_axis_u,
		        dL_dtx_axis_v,
		        dL_dimg_params,
        dL_dgs_trans,
        dL_dgs_quats,
        dL_dgs_scales,
        dL_dgs_reflections,
        dL_dgs_screenspace_points);
}
