// CUDA / cuFFT migration.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Sim.h"
#include "CudaCheck.h"

// ============================================================================
// Constant memory: the 4 sampled water depths used by the eWave Airy solver.
// ============================================================================
__constant__ float c_Depth[DEPTH_NUM];

// One block of GRIDRESOLUTION threads is enough since N = 256.
static constexpr int kBlock = 256;
static constexpr int kGrid  = (GRIDRESOLUTION + kBlock - 1) / kBlock;

// ============================================================================
// Device-side helper functions.
// ============================================================================
__device__ __forceinline__ int dev_clamp_idx(int x)
{
    return x < 0 ? 0 : (x >= GRIDRESOLUTION ? GRIDRESOLUTION - 1 : x);
}

// 1D: 0.5 of the source-cell content per timestep guarantees volume conservation.
__device__ __forceinline__ float Limit_flow_rate_d(float fr, float hL, float hR)
{
    if (fr >= 0.f) return fminf(fr,  0.5f * hL * (float)GRIDCELLSIZE / TIMESTEP);
    else           return fmaxf(fr, -0.5f * hR * (float)GRIDCELLSIZE / TIMESTEP);
}

// 1D: 0.5 guarantees the CFL condition.
__device__ __forceinline__ float LimitVelocity_d(float v)
{
    if (v >= 0.f) return fminf(v,  0.5f * (float)GRIDCELLSIZE / TIMESTEP);
    else          return fmaxf(v, -0.5f * (float)GRIDCELLSIZE / TIMESTEP);
}

__device__ __forceinline__ bool StopFlowOnTerrainBoundary_d(int x, const float* h, const float* terrain)
{
    const float epsilon = 0.01f;
    int xp1 = dev_clamp_idx(x + 1);
    if ((h[x]   <= epsilon) && (terrain[x]   >= terrain[xp1] + h[xp1])) return true;
    if ((h[xp1] <= epsilon) && (terrain[xp1] >  terrain[x]   + h[x]))   return true;
    return false;
}

// Cubic interpolation with Catmull-Rom spline + BFECC-style value limiting.
__device__ __forceinline__ float SampleCubicClamped_d(float samplePos, const float* dataField)
{
    int   id_start = (int)floorf(samplePos) - 1;
    int   id0 = dev_clamp_idx(id_start + 0);
    int   id1 = dev_clamp_idx(id_start + 1);
    int   id2 = dev_clamp_idx(id_start + 2);
    int   id3 = dev_clamp_idx(id_start + 3);
    float fx  = fmaxf(0.f, fminf(1.f, samplePos - floorf(samplePos)));
    float x2  = fx * fx;
    float x3  = x2 * fx;
    const float s = 0.5f;
    float xX = -s * x3 + 2.f * s * x2 - s * fx;
    float xY = (2.f - s) * x3 + (s - 3.f) * x2 + 1.f;
    float xZ = (s - 2.f) * x3 + (3.f - 2.f * s) * x2 + s * fx;
    float xW =  s * x3 - s * x2;
    float out = xX * dataField[id0] + xY * dataField[id1]
              + xZ * dataField[id2] + xW * dataField[id3];
    out = fminf(fmaxf(dataField[id1], dataField[id2]), out);
    out = fmaxf(fminf(dataField[id1], dataField[id2]), out);
    return out;
}

// ============================================================================
// Bulk vs. surface decomposition
// ============================================================================

__global__ void kernel_init_decomp(
    const float* __restrict__ terrain,
    const float* __restrict__ h,
    const float* __restrict__ q,
    float* __restrict__ hbar,
    float* __restrict__ qbar,
    float* __restrict__ alpha_hbar)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    hbar[x] = terrain[x] + h[x];
    qbar[x] = q[x];

    float a = 0.f;
    float maxGround     = fmaxf(terrain[x], terrain[xp1]);
    float minWaterlevel = 0.5f * (terrain[x] + h[x] + terrain[xp1] + h[xp1]);
    if ((h[x] > 0.f) && (h[xp1] > 0.f)) {
        const float sigma_max = 8.f;
        float sigma = fminf(sigma_max, fmaxf(0.f, minWaterlevel - maxGround));
        a = sigma * sigma / (sigma_max * sigma_max);
    }
    float gradient = fabsf(terrain[x] + h[x] - (terrain[xp1] + h[xp1]));
    alpha_hbar[x] = a * expf(-0.01f * gradient * gradient);
}

__global__ void kernel_compute_alpha_qbar(
    const float* __restrict__ alpha_hbar,
    float* __restrict__ alpha_qbar)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    alpha_qbar[x] = 0.5f * (alpha_hbar[xm1] + alpha_hbar[x]);
}

// One explicit diffusion step.
__global__ void kernel_diffusion_step(
    const float* __restrict__ hbar_in,
    const float* __restrict__ qbar_in,
    float* __restrict__ hbar_out,
    float* __restrict__ qbar_out,
    const float* __restrict__ terrain,
    const float* __restrict__ alpha_hbar,
    const float* __restrict__ alpha_qbar)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= GRIDRESOLUTION) return;

    if (i == GRIDRESOLUTION - 1) {
        hbar_out[i] = hbar_in[i];
        qbar_out[i] = qbar_in[i];
        return;
    }
    int ip1 = dev_clamp_idx(i + 1);
    int im1 = dev_clamp_idx(i - 1);

    float hbar_new = hbar_in[i] + 0.48f * (
          alpha_hbar[i]   * (hbar_in[ip1] - hbar_in[i])
        - alpha_hbar[im1] * (hbar_in[i]   - hbar_in[im1])
    );
    hbar_out[i] = fmaxf(terrain[i], hbar_new);

    float qbar_new = qbar_in[i] + 0.48f * (
          alpha_qbar[ip1] * (qbar_in[ip1] - qbar_in[i])
        - alpha_qbar[i]   * (qbar_in[i]   - qbar_in[im1])
    );
    qbar_out[i] = qbar_new;
}

__global__ void kernel_finalize_decomposition(
    const float* __restrict__ terrain,
    const float* __restrict__ h,
    const float* __restrict__ q,
    float* __restrict__ hbar,
    float* __restrict__ htilde,
    float* __restrict__ qbar,
    float* __restrict__ qtilde)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;

    float hb = fmaxf(0.f, hbar[x] - terrain[x]);
    hbar[x]   = hb;
    htilde[x] = h[x] - hb;
    qtilde[x] = q[x] - qbar[x];

    if (StopFlowOnTerrainBoundary_d(x, h, terrain)) {
        qbar[x]   = 0.f;
        qtilde[x] = 0.f;
    }
}

// ============================================================================
// Surface velocity update via eWave.
// ============================================================================

// Pack input arrays into the batched FFT buffer:
//   fwd_buf[0..N)        = 0.5*(htilde + htildeOld)   (real, time-averaged)
//   fwd_buf[N..2N)       = qtilde                     (real)
// Also updates htildeOld with the current htilde for the next step.
// Race-free: each thread reads/writes only its own index.
__global__ void kernel_pack_fwd_input(
    const float* __restrict__ htilde,
    float* __restrict__ htildeOld_inout,
    const float* __restrict__ qtilde,
    cufftComplex* __restrict__ fwd_buf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;

    float ht_curr = htilde[x];
    float ht_old  = htildeOld_inout[x];

    fwd_buf[x].x                  = 0.5f * (ht_curr + ht_old);
    fwd_buf[x].y                  = 0.f;
    fwd_buf[GRIDRESOLUTION + x].x = qtilde[x];
    fwd_buf[GRIDRESOLUTION + x].y = 0.f;

    htildeOld_inout[x] = ht_curr;
}

// Frequency-domain evolution
__global__ void kernel_freq_domain(
    const cufftComplex* __restrict__ fwd_buf,
    cufftComplex* __restrict__ inv_buf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;

    const float Nf = (float)GRIDRESOLUTION;
    float kx = Nf * 0.5f - fabsf(Nf * 0.5f - (float)x);
    float k  = 2.f * PI * fabsf(kx) / Nf / (float)GRIDCELLSIZE;
    float kNonZero = fmaxf(0.01f, k);
    float kS = ((float)x > Nf * 0.5f) ? -k : k;

    // Fourier gradient: htildehat *= -i*kS
    float h_re = fwd_buf[x].x;
    float h_im = fwd_buf[x].y;
    float g_re = -kS * h_im;
    float g_im =  kS * h_re;

    // Half-cell phase shift: e^{i * 0.5 * dx * kS}
    float beta = 0.5f * (float)GRIDCELLSIZE * kS;
    float cb = cosf(beta), sb = sinf(beta);
    float hs_re = cb * g_re - sb * g_im;
    float hs_im = sb * g_re + cb * g_im;

    float q_re = fwd_buf[GRIDRESOLUTION + x].x;
    float q_im = fwd_buf[GRIDRESOLUTION + x].y;

    #pragma unroll
    for (int d = 0; d < DEPTH_NUM; d++) {
        float k2    = fmaxf(0.0001f, 2.f * kx / Nf);                   // k2 in (0,1]
        float omega = sqrtf(GRAVITY * k * tanhf(k * c_Depth[d]));
        omega *= 1.f / sqrtf(2.f / (k2 * PI) * sinf(k2 * PI * 0.5f));   // grid dispersion correction
        float co  = cosf(omega * TIMESTEP);
        float si  = sinf(omega * TIMESTEP);
        float coef = omega / (kNonZero * kNonZero) * si;
        inv_buf[d * GRIDRESOLUTION + x].x = q_re * co - coef * hs_re;
        inv_buf[d * GRIDRESOLUTION + x].y = q_im * co - coef * hs_im;
    }
}

// Pick the surface flow rate corresponding to the local water depth by
// linearly interpolating between the two closest pre-computed depth solutions.
__global__ void kernel_depth_interp_qtilde(
    const cufftComplex* __restrict__ inv_buf,
    const float* __restrict__ hbar,
    float* __restrict__ qtilde)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    float waterDepth = fmaxf(hbar[x], hbar[xp1]);

    int depth1 = 0;
    #pragma unroll
    for (int d = 0; d < DEPTH_NUM; d++)
        if (waterDepth >= c_Depth[d]) depth1 = d;
    int depth2 = depth1 + 1;
    if (depth2 >= DEPTH_NUM) depth2 = DEPTH_NUM - 1;

    float s = 0.f;
    if (depth1 != depth2)
        s = (c_Depth[depth2] - waterDepth) / (c_Depth[depth2] - c_Depth[depth1]);

    float v1 = inv_buf[depth1 * GRIDRESOLUTION + x].x;
    float v2 = inv_buf[depth2 * GRIDRESOLUTION + x].x;
    const float invN = 1.f / (float)GRIDRESOLUTION;
    qtilde[x] = (s * v1 + (1.f - s) * v2) * invN;
}

// ============================================================================
// SWE bulk simulation
// ============================================================================

__global__ void kernel_qbar_to_ubar(
    const float* __restrict__ qbar,
    const float* __restrict__ hbarOld,
    float* __restrict__ ubar)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    float u = qbar[x];
    if (u >= 0.f) u /= fmaxf(0.01f, hbarOld[x]);
    else          u /= fmaxf(0.01f, hbarOld[xp1]);
    ubar[x] = LimitVelocity_d(u);
}

__global__ void kernel_swe_momentum(
    const float* __restrict__ ubar,
    const float* __restrict__ hbar,
    const float* __restrict__ terrain,
    float* __restrict__ ubarNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    int xp1 = dev_clamp_idx(x + 1);
    int xp2 = dev_clamp_idx(x + 2);

    float q_m05 = ubar[xm1];
    if (q_m05 >= 0.f) q_m05 *= hbar[xm1];
    else              q_m05 *= hbar[x];

    float q_p05 = ubar[x];
    if (q_p05 >= 0.f) q_p05 *= hbar[x];
    else              q_p05 *= hbar[xp1];

    float q_p15 = ubar[xp1];
    if (q_p15 >= 0.f) q_p15 *= hbar[xp1];
    else              q_p15 *= hbar[xp2];

    float q_bar_0  = 0.5f * (q_m05 + q_p05);
    float q_bar_p1 = 0.5f * (q_p05 + q_p15);
    float u_star_0  = (q_bar_0  >= 0.f) ? ubar[xm1] : ubar[x];
    float u_star_p1 = (q_bar_p1 >  0.f) ? ubar[x]   : ubar[xp1];

    float uu_x = 2.f / fmaxf(0.01f, hbar[x] + hbar[xp1])
               * ((q_bar_p1 * u_star_p1 - q_bar_0 * u_star_0) / (float)GRIDCELLSIZE
                  - ubar[x] * (q_bar_p1 - q_bar_0) / (float)GRIDCELLSIZE);

    float un = ubar[x] - TIMESTEP * uu_x;
    un += -GRAVITY * TIMESTEP * (terrain[xp1] + hbar[xp1]
                                 - terrain[x]   - hbar[x]) / (float)GRIDCELLSIZE;
    ubarNew[x] = LimitVelocity_d(un);
}

__global__ void kernel_ubar_to_qbar(
    const float* __restrict__ ubarNew,
    const float* __restrict__ hbar,
    float* __restrict__ qbar)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    float u = ubarNew[x];
    if (u >= 0.f) qbar[x] = u * hbar[x];
    else          qbar[x] = u * hbar[xp1];
}

// ============================================================================
// Transport surface flow rate / surface height through bulk velocity
// ============================================================================

__global__ void kernel_advect_qtilde(
    const float* __restrict__ ubar,
    const float* __restrict__ ubarNew,
    const float* __restrict__ h,
    const float* __restrict__ qtilde_in,
    float* __restrict__ qtilde_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    float bulkVelocity = 0.5f * (ubarNew[x] + ubar[x]);
    float val = SampleCubicClamped_d((float)x - TIMESTEP * bulkVelocity, qtilde_in);

    if (((bulkVelocity >= 0.f) && (h[x]   < 0.01f)) ||
        ((bulkVelocity <  0.f) && (h[xp1] < 0.01f)))
        val = 0.f;
    qtilde_out[x] = val;
}

// G = min(-div u, -gamma * div u),  qtilde *= exp(G * dt)
__global__ void kernel_div_ubar_qtilde(
    const float* __restrict__ ubar,
    const float* __restrict__ ubarNew,
    float* __restrict__ qtilde)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    int xp1 = dev_clamp_idx(x + 1);
    float u_m1 = 0.5f * (ubarNew[xm1] + ubar[xm1]);
    float u_p1 = 0.5f * (ubarNew[xp1] + ubar[xp1]);
    float div_ubar = (u_p1 - u_m1) / (2.f * (float)GRIDCELLSIZE);
    if (div_ubar < 0.f) div_ubar *= 0.25f;       // gamma = 1/4 amplification damping
    qtilde[x] *= expf(-div_ubar * TIMESTEP);
}

__global__ void kernel_div_ubar_htilde(
    const float* __restrict__ ubarNew,
    float* __restrict__ htilde)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    float div_ubar = (ubarNew[x] - ubarNew[xm1]) / (float)GRIDCELLSIZE;
    if (div_ubar < 0.f) div_ubar *= 0.25f;
    htilde[x] *= expf(-div_ubar * TIMESTEP);
}

// ============================================================================
// Bulk-advected surface displacement -> flux update of h 
// ============================================================================

__global__ void kernel_compute_advectHFR(
    const float* __restrict__ ubarNew,
    const float* __restrict__ htilde,
    float* __restrict__ advectHFR)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    advectHFR[x] = ubarNew[x]
                 * SampleCubicClamped_d((float)x + 0.5f - 0.5f * TIMESTEP * ubarNew[x], htilde);
}

__global__ void kernel_h_update_from_advectHFR(
    const float* __restrict__ advectHFR,
    const float* __restrict__ h_in,
    const float* __restrict__ terrain,
    float* __restrict__ h_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    int xp1 = dev_clamp_idx(x + 1);

    float q_l = Limit_flow_rate_d(advectHFR[xm1], h_in[xm1], h_in[x]);
    float q_r = Limit_flow_rate_d(advectHFR[x],   h_in[x],   h_in[xp1]);
    if (((h_in[xm1] == 0.f) && (h_in[x] == 0.f))
        || StopFlowOnTerrainBoundary_d(xm1, h_in, terrain))
        q_l = 0.f;
    if (((h_in[x] == 0.f) && (h_in[xp1] == 0.f))
        || StopFlowOnTerrainBoundary_d(x, h_in, terrain))
        q_r = 0.f;

    h_out[x] = fmaxf(0.f, h_in[x] - TIMESTEP / (float)GRIDCELLSIZE * (q_r - q_l));
}

// ============================================================================
// Recombine flow rates and final height integration.
// ============================================================================

__global__ void kernel_recombine_q(
    const float* __restrict__ qbar,
    const float* __restrict__ qtilde,
    const float* __restrict__ h,
    const float* __restrict__ terrain,
    float* __restrict__ q)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);

    float qx = Limit_flow_rate_d(qbar[x] + qtilde[x], h[x], h[xp1]);
    if (StopFlowOnTerrainBoundary_d(x, h, terrain) || (x == 0) || (x >= GRIDRESOLUTION - 2))
        qx = 0.f;
    q[x] = qx;
}

__global__ void kernel_height_integration(
    const float* __restrict__ q,
    float* __restrict__ h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xm1 = dev_clamp_idx(x - 1);
    h[x] = fmaxf(0.f, h[x] + TIMESTEP * -(q[x] - q[xm1]) / (float)GRIDCELLSIZE);
}

__global__ void kernel_q_final_limit(
    const float* __restrict__ h,
    float* __restrict__ q)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= GRIDRESOLUTION) return;
    int xp1 = dev_clamp_idx(x + 1);
    q[x] = Limit_flow_rate_d(q[x], h[x], h[xp1]);
}

// ============================================================================
// Sim member functions
// ============================================================================

Sim::Sim()
    : terrain_d(nullptr), h_d(nullptr), q_d(nullptr),
      hbarOld_d(nullptr), htildeOld_d(nullptr),
      hbar_d(nullptr), qbar_d(nullptr),
      htilde_d(nullptr), qtilde_d(nullptr),
      alpha_hbar_d(nullptr), alpha_qbar_d(nullptr),
      hbar_dummy_d(nullptr), qbar_dummy_d(nullptr),
      ubar_d(nullptr), ubarNew_d(nullptr),
      qtilde_dummy_d(nullptr), advectHFR_d(nullptr),
      h_dummy_d(nullptr),
      fwd_buf_d(nullptr), inv_buf_d(nullptr),
      plan_fwd(0), plan_inv(0), time(0.f)
{
    const size_t bytesF = (size_t)GRIDRESOLUTION * sizeof(float);

    CUDA_CHECK(cudaMalloc(&terrain_d,      bytesF));
    CUDA_CHECK(cudaMalloc(&h_d,            bytesF));
    CUDA_CHECK(cudaMalloc(&q_d,            bytesF));
    CUDA_CHECK(cudaMalloc(&hbarOld_d,      bytesF));
    CUDA_CHECK(cudaMalloc(&htildeOld_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&hbar_d,         bytesF));
    CUDA_CHECK(cudaMalloc(&qbar_d,         bytesF));
    CUDA_CHECK(cudaMalloc(&htilde_d,       bytesF));
    CUDA_CHECK(cudaMalloc(&qtilde_d,       bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_hbar_d,   bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_qbar_d,   bytesF));
    CUDA_CHECK(cudaMalloc(&hbar_dummy_d,   bytesF));
    CUDA_CHECK(cudaMalloc(&qbar_dummy_d,   bytesF));
    CUDA_CHECK(cudaMalloc(&ubar_d,         bytesF));
    CUDA_CHECK(cudaMalloc(&ubarNew_d,      bytesF));
    CUDA_CHECK(cudaMalloc(&qtilde_dummy_d, bytesF));
    CUDA_CHECK(cudaMalloc(&advectHFR_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&h_dummy_d,      bytesF));

    CUDA_CHECK(cudaMalloc(&fwd_buf_d, 2 * (size_t)GRIDRESOLUTION * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&inv_buf_d, (size_t)DEPTH_NUM * (size_t)GRIDRESOLUTION * sizeof(cufftComplex)));

    int n[1] = { GRIDRESOLUTION };
    CUFFT_CHECK(cufftPlanMany(&plan_fwd, 1, n,
                              NULL, 1, GRIDRESOLUTION,
                              NULL, 1, GRIDRESOLUTION,
                              CUFFT_C2C, /*batch=*/ 2));
    CUFFT_CHECK(cufftPlanMany(&plan_inv, 1, n,
                              NULL, 1, GRIDRESOLUTION,
                              NULL, 1, GRIDRESOLUTION,
                              CUFFT_C2C, /*batch=*/ DEPTH_NUM));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Depth, Depth, DEPTH_NUM * sizeof(float)));

    ResetTerrain(1);
    ResetWater(2, 0.f);
}

int Sim::Release(void)
{
    if (plan_fwd) cufftDestroy(plan_fwd);
    if (plan_inv) cufftDestroy(plan_inv);
    plan_fwd = plan_inv = 0;

    cudaFree(terrain_d);     terrain_d     = nullptr;
    cudaFree(h_d);           h_d           = nullptr;
    cudaFree(q_d);           q_d           = nullptr;
    cudaFree(hbarOld_d);     hbarOld_d     = nullptr;
    cudaFree(htildeOld_d);   htildeOld_d   = nullptr;
    cudaFree(hbar_d);        hbar_d        = nullptr;
    cudaFree(qbar_d);        qbar_d        = nullptr;
    cudaFree(htilde_d);      htilde_d      = nullptr;
    cudaFree(qtilde_d);      qtilde_d      = nullptr;
    cudaFree(alpha_hbar_d);  alpha_hbar_d  = nullptr;
    cudaFree(alpha_qbar_d);  alpha_qbar_d  = nullptr;
    cudaFree(hbar_dummy_d);  hbar_dummy_d  = nullptr;
    cudaFree(qbar_dummy_d);  qbar_dummy_d  = nullptr;
    cudaFree(ubar_d);        ubar_d        = nullptr;
    cudaFree(ubarNew_d);     ubarNew_d     = nullptr;
    cudaFree(qtilde_dummy_d); qtilde_dummy_d = nullptr;
    cudaFree(advectHFR_d);   advectHFR_d   = nullptr;
    cudaFree(h_dummy_d);     h_dummy_d     = nullptr;
    cudaFree(fwd_buf_d);     fwd_buf_d     = nullptr;
    cudaFree(inv_buf_d);     inv_buf_d     = nullptr;
    return 0;
}

// ----------------------------------------------------------------------------
// Init functions  (host shadow construction + H2D upload)
// ----------------------------------------------------------------------------

// type: 0=flat, 1=hill
void Sim::ResetTerrain(int type)
{
    for (int x = 0; x < GRIDRESOLUTION; x++) {
        if (type == 0)
            terrain[x] = -fabsf((float)TERRAIN_HEIGHT_SHIFT_INIT);
        else if (type == 1)
            terrain[x] = (-1.f + 0.1f
                          + 0.1f  * (float)x / GRIDRESOLUTION
                          + 0.03f * sinf(20.f  * (float)x / GRIDRESOLUTION)
                          + 0.9f  * sinf(2.5f  * (float)x / GRIDRESOLUTION))
                         * fabsf((float)TERRAIN_HEIGHT_SHIFT_INIT);
    }
    terrain[0]                  = 1.8f * fabsf((float)TERRAIN_HEIGHT_SHIFT_INIT);
    terrain[GRIDRESOLUTION - 1] = 1.8f * fabsf((float)TERRAIN_HEIGHT_SHIFT_INIT);

    CUDA_CHECK(cudaMemcpy(terrain_d, terrain,
                          GRIDRESOLUTION * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// type: 0=constant level, 1=dam break, 2=sloped, 3=flat with cosine waves
void Sim::ResetWater(int type, float level)
{
    for (int x = 0; x < GRIDRESOLUTION; x++) {
        if (type == 0)
            h[x] = fmaxf(0.f, level - terrain[x]);
        if (type == 1) {
            if (x <= GRIDRESOLUTION / 2) h[x] = 0.f;
            else                         h[x] = fmaxf(0.f, level - terrain[x]);
        }
        if (type == 2) {
            float t = -0.5f + (float)x / GRIDRESOLUTION;
            h[x] = fmaxf(0.f, level
                         + (2.f * t * fabsf(t)) * fabsf(0.5f * (float)TERRAIN_HEIGHT_SHIFT_INIT)
                         - terrain[x]);
        }
        if (type == 3) {
            const float lambda = 10.f;
            h[x] = fmaxf(0.f, level + 0.5f * cosf(2.f * PI * ((float)x / lambda)) - terrain[x]);
        }

        hbar[x]   = h[x];
        qbar[x]   = 0.f;
        htilde[x] = 0.f;
        qtilde[x] = 0.f;
        q[x]      = 0.f;
    }
    h[0]                  = 0.f;
    h[GRIDRESOLUTION - 1] = 0.f;
    hbar[0]                  = 0.f;
    hbar[GRIDRESOLUTION - 1] = 0.f;

    const size_t bytesF = (size_t)GRIDRESOLUTION * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_d,         h,      bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hbar_d,      hbar,   bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hbarOld_d,   hbar,   bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(htildeOld_d, 0,      bytesF));
    CUDA_CHECK(cudaMemset(htilde_d,    0,      bytesF));
    CUDA_CHECK(cudaMemset(qbar_d,      0,      bytesF));
    CUDA_CHECK(cudaMemset(q_d,         0,      bytesF));
    CUDA_CHECK(cudaMemset(qtilde_d,    0,      bytesF));

    time = 0.f;
}

// xCoord and size in (0..1), factor determines how much to add/subtract
void Sim::EditWaterLocal(float xCoord, float size, float factor)
{
    // Round-trip readback / modify / upload (small data, simple & robust).
    CUDA_CHECK(cudaMemcpy(h, h_d,
                          GRIDRESOLUTION * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int x = 0; x < GRIDRESOLUTION; x++) {
        if (fabsf((float)x / GRIDRESOLUTION - xCoord) < size)
            h[x] = fmaxf(0.f, h[x] + factor * 1.f);
    }
    h[0]                  = 0.f;
    h[GRIDRESOLUTION - 1] = 0.f;
    CUDA_CHECK(cudaMemcpy(h_d, h,
                          GRIDRESOLUTION * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Sim::SyncToHost()
{
    const size_t bytesF = (size_t)GRIDRESOLUTION * sizeof(float);
    CUDA_CHECK(cudaMemcpy(terrain, terrain_d, bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h,       h_d,       bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q,       q_d,       bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbar,    hbar_d,    bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qbar,    qbar_d,    bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(htilde,  htilde_d,  bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qtilde,  qtilde_d,  bytesF, cudaMemcpyDeviceToHost));
}

// ============================================================================
// SimStep — the heart of the simulation.
// Each block of kernels below corresponds to a comment-delimited section of
// the original CPU Sim.cpp.
// ============================================================================
void Sim::SimStep(bool SWEonly)
{
    const size_t bytesF = (size_t)GRIDRESOLUTION * sizeof(float);

    // ------------------------------------------------------------------
    // Bulk vs. surface decomposition
    // ------------------------------------------------------------------
    kernel_init_decomp        <<<kGrid, kBlock>>>(terrain_d, h_d, q_d,
                                                  hbar_d, qbar_d, alpha_hbar_d);
    CUDA_CHECK_LAST();
    kernel_compute_alpha_qbar <<<kGrid, kBlock>>>(alpha_hbar_d, alpha_qbar_d);
    CUDA_CHECK_LAST();

    if (!SWEonly) {
        // Seed the dummy buffers with the current state so that index N-1
        // (which the diffusion kernel only copies, never updates) starts
        // with the correct value in both ping-pong slots.
        CUDA_CHECK(cudaMemcpyAsync(hbar_dummy_d, hbar_d, bytesF, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(qbar_dummy_d, qbar_d, bytesF, cudaMemcpyDeviceToDevice));

        float* h_in  = hbar_d;       float* h_out = hbar_dummy_d;
        float* q_in  = qbar_d;       float* q_out = qbar_dummy_d;
        for (int j = 0; j < 64; j++) {
            kernel_diffusion_step<<<kGrid, kBlock>>>(h_in, q_in, h_out, q_out,
                                                     terrain_d, alpha_hbar_d, alpha_qbar_d);
            std::swap(h_in, h_out);
            std::swap(q_in, q_out);
        }
        // After 64 swaps h_in == hbar_d again, so latest data is in hbar_d
        // -- but be defensive in case the iteration count ever changes.
        if (h_in != hbar_d) {
            CUDA_CHECK(cudaMemcpyAsync(hbar_d, h_in, bytesF, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(qbar_d, q_in, bytesF, cudaMemcpyDeviceToDevice));
        }
    }

    kernel_finalize_decomposition<<<kGrid, kBlock>>>(terrain_d, h_d, q_d,
                                                     hbar_d, htilde_d,
                                                     qbar_d, qtilde_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // Surface velocity update via eWave
    // ------------------------------------------------------------------
    kernel_pack_fwd_input<<<kGrid, kBlock>>>(htilde_d, htildeOld_d, qtilde_d, fwd_buf_d);
    CUDA_CHECK_LAST();

    CUFFT_CHECK(cufftExecC2C(plan_fwd, fwd_buf_d, fwd_buf_d, CUFFT_FORWARD));

    kernel_freq_domain<<<kGrid, kBlock>>>(fwd_buf_d, inv_buf_d);
    CUDA_CHECK_LAST();

    CUFFT_CHECK(cufftExecC2C(plan_inv, inv_buf_d, inv_buf_d, CUFFT_INVERSE));

    kernel_depth_interp_qtilde<<<kGrid, kBlock>>>(inv_buf_d, hbar_d, qtilde_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // SWE bulk simulation
    // ------------------------------------------------------------------
    kernel_qbar_to_ubar<<<kGrid, kBlock>>>(qbar_d, hbarOld_d, ubar_d);
    CUDA_CHECK_LAST();

    // store current hbar for next timestep (replaces CPU memcpy)
    CUDA_CHECK(cudaMemcpyAsync(hbarOld_d, hbar_d, bytesF, cudaMemcpyDeviceToDevice));

    kernel_swe_momentum<<<kGrid, kBlock>>>(ubar_d, hbar_d, terrain_d, ubarNew_d);
    CUDA_CHECK_LAST();
    kernel_ubar_to_qbar<<<kGrid, kBlock>>>(ubarNew_d, hbar_d, qbar_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // Advect qtilde + amplification damping for qtilde / htilde
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(qtilde_dummy_d, qtilde_d, bytesF, cudaMemcpyDeviceToDevice));
    kernel_advect_qtilde   <<<kGrid, kBlock>>>(ubar_d, ubarNew_d, h_d,
                                               qtilde_dummy_d, qtilde_d);
    CUDA_CHECK_LAST();
    kernel_div_ubar_qtilde <<<kGrid, kBlock>>>(ubar_d, ubarNew_d, qtilde_d);
    CUDA_CHECK_LAST();
    kernel_div_ubar_htilde <<<kGrid, kBlock>>>(ubarNew_d, htilde_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // Bulk-advected surface displacement -> flux update of h
    // ------------------------------------------------------------------
    kernel_compute_advectHFR<<<kGrid, kBlock>>>(ubarNew_d, htilde_d, advectHFR_d);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(h_dummy_d, h_d, bytesF, cudaMemcpyDeviceToDevice));
    kernel_h_update_from_advectHFR<<<kGrid, kBlock>>>(advectHFR_d, h_dummy_d, terrain_d, h_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // recombine + height integration  + final limit
    // ------------------------------------------------------------------
    kernel_recombine_q       <<<kGrid, kBlock>>>(qbar_d, qtilde_d, h_d, terrain_d, q_d);
    CUDA_CHECK_LAST();
    kernel_height_integration<<<kGrid, kBlock>>>(q_d, h_d);
    CUDA_CHECK_LAST();
    kernel_q_final_limit     <<<kGrid, kBlock>>>(h_d, q_d);
    CUDA_CHECK_LAST();

    time += TIMESTEP;

    // Make the host shadow arrays usable to the renderer immediately after
    // every step.  Cost is 7*N*sizeof(float) ~= 7 KB per step; negligible.
    SyncToHost();
}
