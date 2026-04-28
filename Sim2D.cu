// CUDA / cuFFT 2D extension of the 1D dispersive SWE simulator.
// Mirrors Algorithm 1 of Jeschke & Wojtan, "Generalizing Shallow Water
// Simulations with Dispersive Surface Waves" (SIGGRAPH 2023) on an N x N
// MAC (staggered) grid. Each section below corresponds 1:1 to the
// comment-delimited blocks of the 1D Sim.cu.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Sim2D.h"
#include "CudaCheck.h"

// ============================================================================
// Constant memory: the 4 sampled water depths used by the eWave Airy solver.
// ============================================================================
__constant__ float c_Depth_2D[DEPTH_NUM_2D];

// 2D launch geometry: 16x16 thread blocks tile the N x N domain.
static const dim3 kBlock(16, 16);
static const dim3 kGrid((GRIDRESOLUTION_2D + 15) / 16,
                       (GRIDRESOLUTION_2D + 15) / 16);

// ============================================================================
// Device-side helper functions (single-precision counterparts of the 1D
// inline host helpers).
// ============================================================================
__device__ __forceinline__ int clamp_x(int i)
{
    return i < 0 ? 0 : (i >= GRIDRESOLUTION_2D ? GRIDRESOLUTION_2D - 1 : i);
}

__device__ __forceinline__ int clamp_y(int j)
{
    return j < 0 ? 0 : (j >= GRIDRESOLUTION_2D ? GRIDRESOLUTION_2D - 1 : j);
}

__device__ __forceinline__ int idx2(int i, int j)
{
    return j * GRIDRESOLUTION_2D + i;
}

// 2D analogue of 1D Limit_flow_rate: 0.5 of the source-cell content per
// timestep guarantees volume conservation along the relevant axis.
__device__ __forceinline__ float Limit_flow_rate_d(float fr, float hL, float hR)
{
    if (fr >= 0.f) return fminf(fr,  0.5f * hL * (float)GRIDCELLSIZE_2D / TIMESTEP_2D);
    else           return fmaxf(fr, -0.5f * hR * (float)GRIDCELLSIZE_2D / TIMESTEP_2D);
}

__device__ __forceinline__ float LimitVelocity_d(float v)
{
    if (v >= 0.f) return fminf(v,  0.5f * (float)GRIDCELLSIZE_2D / TIMESTEP_2D);
    else          return fmaxf(v, -0.5f * (float)GRIDCELLSIZE_2D / TIMESTEP_2D);
}

// Stop flow across the right-face of cell (i, j) when there is essentially
// no water on one side AND the dry side's terrain rises above the wet side's
// total water level.  Mirrors 1D StopFlowOnTerrainBoundary.
__device__ __forceinline__ bool StopFlowOnTerrainBoundary_x_d(
    int i, int j, const float* h, const float* terrain)
{
    const float epsilon = 0.01f;
    int ip1 = clamp_x(i + 1);
    int c   = idx2(i,   j);
    int cxp = idx2(ip1, j);
    if ((h[c]   <= epsilon) && (terrain[c]   >= terrain[cxp] + h[cxp])) return true;
    if ((h[cxp] <= epsilon) && (terrain[cxp] >  terrain[c]   + h[c]))   return true;
    return false;
}

__device__ __forceinline__ bool StopFlowOnTerrainBoundary_y_d(
    int i, int j, const float* h, const float* terrain)
{
    const float epsilon = 0.01f;
    int jp1 = clamp_y(j + 1);
    int c   = idx2(i, j);
    int cyp = idx2(i, jp1);
    if ((h[c]   <= epsilon) && (terrain[c]   >= terrain[cyp] + h[cyp])) return true;
    if ((h[cyp] <= epsilon) && (terrain[cyp] >  terrain[c]   + h[c]))   return true;
    return false;
}

// Catmull-Rom 4-tap weights (s = 0.5).
__device__ __forceinline__ void crom_weights_d(float t, float& w0, float& w1, float& w2, float& w3)
{
    const float s = 0.5f;
    float t2 = t * t;
    float t3 = t2 * t;
    w0 = -s * t3 + 2.f * s * t2 - s * t;
    w1 = (2.f - s) * t3 + (s - 3.f) * t2 + 1.f;
    w2 = (s - 2.f) * t3 + (3.f - 2.f * s) * t2 + s * t;
    w3 =  s * t3 - s * t2;
}

// 2D bicubic Catmull-Rom sampling with BFECC-style 2x2 inner-block clamping.
__device__ __forceinline__ float SampleCubicClamped2D_d(float sx, float sy, const float* field)
{
    int ix0 = (int)floorf(sx) - 1;
    int iy0 = (int)floorf(sy) - 1;
    float fx = sx - floorf(sx);
    float fy = sy - floorf(sy);
    fx = fmaxf(0.f, fminf(1.f, fx));
    fy = fmaxf(0.f, fminf(1.f, fy));

    float wx0, wx1, wx2, wx3, wy0, wy1, wy2, wy3;
    crom_weights_d(fx, wx0, wx1, wx2, wx3);
    crom_weights_d(fy, wy0, wy1, wy2, wy3);

    int i0 = clamp_x(ix0 + 0);
    int i1 = clamp_x(ix0 + 1);
    int i2 = clamp_x(ix0 + 2);
    int i3 = clamp_x(ix0 + 3);

    float row[4];
    #pragma unroll
    for (int dy = 0; dy < 4; dy++) {
        int jr = clamp_y(iy0 + dy);
        row[dy] = wx0 * field[idx2(i0, jr)]
                + wx1 * field[idx2(i1, jr)]
                + wx2 * field[idx2(i2, jr)]
                + wx3 * field[idx2(i3, jr)];
    }
    float out = wy0 * row[0] + wy1 * row[1] + wy2 * row[2] + wy3 * row[3];

    int j1 = clamp_y(iy0 + 1);
    int j2 = clamp_y(iy0 + 2);
    float v00 = field[idx2(i1, j1)];
    float v10 = field[idx2(i2, j1)];
    float v01 = field[idx2(i1, j2)];
    float v11 = field[idx2(i2, j2)];
    float lo = fminf(fminf(v00, v10), fminf(v01, v11));
    float hi = fmaxf(fmaxf(v00, v10), fmaxf(v01, v11));
    return fmaxf(lo, fminf(hi, out));
}

// ============================================================================
// 4.1  Bulk vs. surface decomposition (paper §4.1, Eq. 11–13)
// ============================================================================

__global__ void kernel_init_decomp(
    const float* __restrict__ terrain,
    const float* __restrict__ h,
    const float* __restrict__ qx,
    const float* __restrict__ qy,
    float* __restrict__ hbar,
    float* __restrict__ qbarx,
    float* __restrict__ qbary,
    float* __restrict__ alpha_x,
    float* __restrict__ alpha_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);
    int cyp = idx2(i, clamp_y(j + 1));

    hbar [c] = terrain[c] + h[c];
    qbarx[c] = qx[c];
    qbary[c] = qy[c];

    // alpha_x at right face of (i, j): conductance between cells (i,j) and (i+1,j).
    {
        float a = 0.f;
        float maxGround     = fmaxf(terrain[c], terrain[cxp]);
        float minWaterlevel = 0.5f * (terrain[c] + h[c] + terrain[cxp] + h[cxp]);
        if ((h[c] > 0.f) && (h[cxp] > 0.f)) {
            const float sigma_max = 8.f;
            float sigma = fminf(sigma_max, fmaxf(0.f, minWaterlevel - maxGround));
            a = sigma * sigma / (sigma_max * sigma_max);
        }
        float gradient = fabsf((terrain[c] + h[c]) - (terrain[cxp] + h[cxp]));
        alpha_x[c] = a * expf(-0.01f * gradient * gradient);
    }
    // alpha_y at top face of (i, j): conductance between cells (i,j) and (i,j+1).
    {
        float a = 0.f;
        float maxGround     = fmaxf(terrain[c], terrain[cyp]);
        float minWaterlevel = 0.5f * (terrain[c] + h[c] + terrain[cyp] + h[cyp]);
        if ((h[c] > 0.f) && (h[cyp] > 0.f)) {
            const float sigma_max = 8.f;
            float sigma = fminf(sigma_max, fmaxf(0.f, minWaterlevel - maxGround));
            a = sigma * sigma / (sigma_max * sigma_max);
        }
        float gradient = fabsf((terrain[c] + h[c]) - (terrain[cyp] + h[cyp]));
        alpha_y[c] = a * expf(-0.01f * gradient * gradient);
    }
}

// Cell-centered averaged conductances used for the principal-direction bond
// in the 5-point qbarx/qbary diffusion stencils (1D code stored alpha_qbar
// at cell centers as well). Cross-direction bonds are computed inline.
__global__ void kernel_compute_alpha_qbar(
    const float* __restrict__ alpha_x,
    const float* __restrict__ alpha_y,
    float* __restrict__ alpha_qbarx,
    float* __restrict__ alpha_qbary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxm = idx2(clamp_x(i - 1), j);
    int cym = idx2(i, clamp_y(j - 1));

    alpha_qbarx[c] = 0.5f * (alpha_x[cxm] + alpha_x[c]);
    alpha_qbary[c] = 0.5f * (alpha_y[cym] + alpha_y[c]);
}

// One explicit FTCS diffusion step. Coefficient 0.24 < 0.25 = 2D von Neumann
// upper bound. Last row/column simply copies its input to the output, mirroring
// the 1D code's choice for the last index.
__global__ void kernel_diffusion_step(
    const float* __restrict__ hbar_in,
    const float* __restrict__ qbarx_in,
    const float* __restrict__ qbary_in,
    float* __restrict__ hbar_out,
    float* __restrict__ qbarx_out,
    float* __restrict__ qbary_out,
    const float* __restrict__ terrain,
    const float* __restrict__ alpha_x,
    const float* __restrict__ alpha_y,
    const float* __restrict__ alpha_qbarx,
    const float* __restrict__ alpha_qbary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c = idx2(i, j);

    if ((i == GRIDRESOLUTION_2D - 1) || (j == GRIDRESOLUTION_2D - 1)) {
        hbar_out [c] = hbar_in [c];
        qbarx_out[c] = qbarx_in[c];
        qbary_out[c] = qbary_in[c];
        return;
    }

    int cxp = idx2(i + 1, j);
    int cxm = idx2(clamp_x(i - 1), j);
    int cyp = idx2(i, j + 1);
    int cym = idx2(i, clamp_y(j - 1));

    const float K = 0.24f;

    // hbar (cell-centered) -- isotropic 5-point Laplacian with face alphas.
    {
        float xR = alpha_x[c]   * (hbar_in[cxp] - hbar_in[c]);
        float xL = alpha_x[cxm] * (hbar_in[c]   - hbar_in[cxm]);
        float yT = alpha_y[c]   * (hbar_in[cyp] - hbar_in[c]);
        float yB = alpha_y[cym] * (hbar_in[c]   - hbar_in[cym]);
        float new_h = hbar_in[c] + K * ((xR - xL) + (yT - yB));
        hbar_out[c] = fmaxf(terrain[c], new_h);
    }

    // qbarx (right-face of (i,j)) -- 5-point stencil.
    {
        // x-bonds: principal direction, conductances live at the two adjacent cell centers.
        float aXR = alpha_qbarx[cxp];
        float aXL = alpha_qbarx[c];
        float xR  = aXR * (qbarx_in[cxp] - qbarx_in[c]);
        float xL  = aXL * (qbarx_in[c]   - qbarx_in[cxm]);
        // y-bonds: at corners (i+1, j+1) and (i+1, j) -- average of the two y-face alphas.
        int cxp_ym = idx2(i + 1, clamp_y(j - 1));
        float aYT = 0.5f * (alpha_y[c]   + alpha_y[cxp]);
        float aYB = 0.5f * (alpha_y[cym] + alpha_y[cxp_ym]);
        float yT  = aYT * (qbarx_in[cyp] - qbarx_in[c]);
        float yB  = aYB * (qbarx_in[c]   - qbarx_in[cym]);
        qbarx_out[c] = qbarx_in[c] + K * ((xR - xL) + (yT - yB));
    }

    // qbary (top-face of (i,j)) -- 5-point stencil.
    {
        // y-bonds: principal direction.
        float aYT = alpha_qbary[cyp];
        float aYB = alpha_qbary[c];
        float yT  = aYT * (qbary_in[cyp] - qbary_in[c]);
        float yB  = aYB * (qbary_in[c]   - qbary_in[cym]);
        // x-bonds: at corners (i+1, j+1) and (i, j+1).
        int cxm_yp = idx2(clamp_x(i - 1), j + 1);
        float aXR = 0.5f * (alpha_x[c]   + alpha_x[cyp]);
        float aXL = 0.5f * (alpha_x[cxm] + alpha_x[cxm_yp]);
        float xR  = aXR * (qbary_in[cxp] - qbary_in[c]);
        float xL  = aXL * (qbary_in[c]   - qbary_in[cxm]);
        qbary_out[c] = qbary_in[c] + K * ((xR - xL) + (yT - yB));
    }
}

__global__ void kernel_finalize_decomposition(
    const float* __restrict__ terrain,
    const float* __restrict__ h,
    const float* __restrict__ qx,
    const float* __restrict__ qy,
    float* __restrict__ hbar,
    float* __restrict__ htilde,
    float* __restrict__ qbarx,
    float* __restrict__ qbary,
    float* __restrict__ qtildex,
    float* __restrict__ qtildey)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c = idx2(i, j);

    float hb = fmaxf(0.f, hbar[c] - terrain[c]);
    hbar   [c] = hb;
    htilde [c] = h[c] - hb;
    qtildex[c] = qx[c] - qbarx[c];
    qtildey[c] = qy[c] - qbary[c];

    if (StopFlowOnTerrainBoundary_x_d(i, j, h, terrain)) {
        qbarx  [c] = 0.f;
        qtildex[c] = 0.f;
    }
    if (StopFlowOnTerrainBoundary_y_d(i, j, h, terrain)) {
        qbary  [c] = 0.f;
        qtildey[c] = 0.f;
    }
}

// ============================================================================
// 4.2  Surface velocity update via eWave (paper §4.3, Algorithm 2)
// ============================================================================

// Pack the 3 forward-FFT input planes:
//   plane 0  = 0.5 * (htilde + htildeOld)          (real, time-averaged)
//   plane 1  = qtildex                              (real)
//   plane 2  = qtildey                              (real)
// Also updates htildeOld with the current htilde for the next step.
__global__ void kernel_pack_fwd_input(
    const float* __restrict__ htilde,
    float* __restrict__ htildeOld_inout,
    const float* __restrict__ qtildex,
    const float* __restrict__ qtildey,
    cufftComplex* __restrict__ fwd_buf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c = idx2(i, j);

    float ht_curr = htilde[c];
    float ht_old  = htildeOld_inout[c];

    fwd_buf[c].x                            = 0.5f * (ht_curr + ht_old);
    fwd_buf[c].y                            = 0.f;
    fwd_buf[GRID_N2_2D + c].x               = qtildex[c];
    fwd_buf[GRID_N2_2D + c].y               = 0.f;
    fwd_buf[2 * GRID_N2_2D + c].x           = qtildey[c];
    fwd_buf[2 * GRID_N2_2D + c].y           = 0.f;

    htildeOld_inout[c] = ht_curr;
}

// 2D frequency-domain evolution (Algorithm 2 core).  For each (kx, ky):
//   1. recover signed kSx, kSy from the FFT bin index
//   2. multiply htilde_hat by -i*kSx / -i*kSy (Fourier gradient) and apply
//      a half-cell phase shift along the corresponding axis
//   3. for each of DEPTH_NUM_2D water depths, evaluate
//      omega = sqrt(g*|k|*tanh(|k|*h)) * tensor-product grid dispersion correction
//   4. write q_hat^{t+dt} = cos(w*dt)*q_hat - (w/|k|^2)*sin(w*dt)*grad_h_hat
//      into the corresponding batch slot in inv_buf (qx in slots 0..D-1,
//      qy in slots D..2D-1).
__global__ void kernel_freq_domain_2d(
    const cufftComplex* __restrict__ fwd_buf,
    cufftComplex* __restrict__ inv_buf)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= GRIDRESOLUTION_2D || iy >= GRIDRESOLUTION_2D) return;
    int c  = idx2(ix, iy);

    const float Nf = (float)GRIDRESOLUTION_2D;
    const float dx = (float)GRIDCELLSIZE_2D;

    float kxBin = Nf * 0.5f - fabsf(Nf * 0.5f - (float)ix);
    float kyBin = Nf * 0.5f - fabsf(Nf * 0.5f - (float)iy);
    float kx_phys = 2.f * PI * kxBin / Nf / dx;
    float ky_phys = 2.f * PI * kyBin / Nf / dx;
    float kSx = ((float)ix > Nf * 0.5f) ? -kx_phys : kx_phys;
    float kSy = ((float)iy > Nf * 0.5f) ? -ky_phys : ky_phys;

    float k_mag2 = kx_phys * kx_phys + ky_phys * ky_phys;
    float k_mag  = sqrtf(k_mag2);
    float kNonZero2 = fmaxf(1e-4f, k_mag2);

    float k2x = fmaxf(0.0001f, 2.f * kxBin / Nf);
    float k2y = fmaxf(0.0001f, 2.f * kyBin / Nf);
    float corrX = 1.f / sqrtf(2.f / (k2x * PI) * sinf(k2x * PI * 0.5f));
    float corrY = 1.f / sqrtf(2.f / (k2y * PI) * sinf(k2y * PI * 0.5f));

    float h_re  = fwd_buf[c].x;
    float h_im  = fwd_buf[c].y;
    float qx_re = fwd_buf[GRID_N2_2D + c].x;
    float qx_im = fwd_buf[GRID_N2_2D + c].y;
    float qy_re = fwd_buf[2 * GRID_N2_2D + c].x;
    float qy_im = fwd_buf[2 * GRID_N2_2D + c].y;

    // Fourier gradient: multiply htilde_hat by -i*kS, axis by axis.
    float gx_re = -kSx * h_im;
    float gx_im =  kSx * h_re;
    float gy_re = -kSy * h_im;
    float gy_im =  kSy * h_re;

    // Half-cell phase shift along the relevant axis (qx is dx/2 right of h,
    // qy is dy/2 above h).
    float betaX = 0.5f * dx * kSx;
    float cbX = cosf(betaX), sbX = sinf(betaX);
    float gxs_re = cbX * gx_re - sbX * gx_im;
    float gxs_im = sbX * gx_re + cbX * gx_im;

    float betaY = 0.5f * dx * kSy;
    float cbY = cosf(betaY), sbY = sinf(betaY);
    float gys_re = cbY * gy_re - sbY * gy_im;
    float gys_im = sbY * gy_re + cbY * gy_im;

    #pragma unroll
    for (int d = 0; d < DEPTH_NUM_2D; d++) {
        float omega = sqrtf(GRAVITY * k_mag * tanhf(k_mag * c_Depth_2D[d]));
        omega *= corrX * corrY;
        float co   = cosf(omega * TIMESTEP_2D);
        float si   = sinf(omega * TIMESTEP_2D);
        float coef = omega / kNonZero2 * si;
        // qtildex output at depth d
        inv_buf[d * GRID_N2_2D + c].x = qx_re * co - coef * gxs_re;
        inv_buf[d * GRID_N2_2D + c].y = qx_im * co - coef * gxs_im;
        // qtildey output at depth DEPTH_NUM_2D + d
        inv_buf[(DEPTH_NUM_2D + d) * GRID_N2_2D + c].x = qy_re * co - coef * gys_re;
        inv_buf[(DEPTH_NUM_2D + d) * GRID_N2_2D + c].y = qy_im * co - coef * gys_im;
    }
}

// Pick the surface flow rate corresponding to the local water depth by
// linearly interpolating between the two closest pre-computed depth solutions.
// Also performs the cuFFT inverse normalization (multiply by 1/(N*N)) here.
__global__ void kernel_depth_interp_qtildex(
    const cufftComplex* __restrict__ inv_buf,
    const float* __restrict__ hbar,
    float* __restrict__ qtildex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);

    float waterDepth = fmaxf(hbar[c], hbar[cxp]);
    int depth1 = 0;
    #pragma unroll
    for (int d = 0; d < DEPTH_NUM_2D; d++)
        if (waterDepth >= c_Depth_2D[d]) depth1 = d;
    int depth2 = depth1 + 1;
    if (depth2 >= DEPTH_NUM_2D) depth2 = DEPTH_NUM_2D - 1;

    float s = 0.f;
    if (depth1 != depth2)
        s = (c_Depth_2D[depth2] - waterDepth) / (c_Depth_2D[depth2] - c_Depth_2D[depth1]);

    float v1 = inv_buf[depth1 * GRID_N2_2D + c].x;
    float v2 = inv_buf[depth2 * GRID_N2_2D + c].x;
    const float invN = 1.f / (float)GRID_N2_2D;
    qtildex[c] = (s * v1 + (1.f - s) * v2) * invN;
}

__global__ void kernel_depth_interp_qtildey(
    const cufftComplex* __restrict__ inv_buf,
    const float* __restrict__ hbar,
    float* __restrict__ qtildey)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cyp = idx2(i, clamp_y(j + 1));

    float waterDepth = fmaxf(hbar[c], hbar[cyp]);
    int depth1 = 0;
    #pragma unroll
    for (int d = 0; d < DEPTH_NUM_2D; d++)
        if (waterDepth >= c_Depth_2D[d]) depth1 = d;
    int depth2 = depth1 + 1;
    if (depth2 >= DEPTH_NUM_2D) depth2 = DEPTH_NUM_2D - 1;

    float s = 0.f;
    if (depth1 != depth2)
        s = (c_Depth_2D[depth2] - waterDepth) / (c_Depth_2D[depth2] - c_Depth_2D[depth1]);

    int o1 = DEPTH_NUM_2D + depth1;
    int o2 = DEPTH_NUM_2D + depth2;
    float v1 = inv_buf[o1 * GRID_N2_2D + c].x;
    float v2 = inv_buf[o2 * GRID_N2_2D + c].x;
    const float invN = 1.f / (float)GRID_N2_2D;
    qtildey[c] = (s * v1 + (1.f - s) * v2) * invN;
}

// ============================================================================
// 4.3  SWE bulk simulation [Stelling & Duinmeijer 2003] (paper §4.2, App. A)
// ============================================================================

__global__ void kernel_qbar_to_ubar_x(
    const float* __restrict__ qbarx,
    const float* __restrict__ hbarOld,
    float* __restrict__ ubarx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);

    float u = qbarx[c];
    if (u >= 0.f) u /= fmaxf(0.01f, hbarOld[c]);
    else          u /= fmaxf(0.01f, hbarOld[cxp]);
    ubarx[c] = LimitVelocity_d(u);
}

__global__ void kernel_qbar_to_ubar_y(
    const float* __restrict__ qbary,
    const float* __restrict__ hbarOld,
    float* __restrict__ ubary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cyp = idx2(i, clamp_y(j + 1));

    float v = qbary[c];
    if (v >= 0.f) v /= fmaxf(0.01f, hbarOld[c]);
    else          v /= fmaxf(0.01f, hbarOld[cyp]);
    ubary[c] = LimitVelocity_d(v);
}

// Full Stelling-Duinmeijer 2D momentum update for ubarx (x-face).  The
// momentum control volume is centred at face (i+1, j+0.5).
//   u_t = -(1/h_face) * [ d(qx*u*)/dx + d(qy*u*)/dy
//                         - u * (d qx/dx + d qy/dy) ]
//        - g * d(tau + hbar)/dx
__global__ void kernel_swe_momentum_x(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubary,
    const float* __restrict__ hbar,
    const float* __restrict__ terrain,
    float* __restrict__ ubarxNew)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;

    int im1 = clamp_x(i - 1);
    int ip1 = clamp_x(i + 1);
    int ip2 = clamp_x(i + 2);
    int jm1 = clamp_y(j - 1);
    int jp1 = clamp_y(j + 1);

    int c       = idx2(i,   j);
    int cxm     = idx2(im1, j);
    int cxp     = idx2(ip1, j);
    int cxpp    = idx2(ip2, j);
    int cym     = idx2(i,   jm1);
    int cyp     = idx2(i,   jp1);
    int cxp_ym  = idx2(ip1, jm1);
    int cxp_yp  = idx2(ip1, jp1);

    float u_self = ubarx[c];
    float u_xL   = ubarx[cxm];
    float u_xR   = ubarx[cxp];
    float u_yB   = ubarx[cym];
    float u_yT   = ubarx[cyp];

    // x-direction qbarx values at the three faces straddling our momentum cell.
    float qfx_L = (u_xL   >= 0.f) ? u_xL   * hbar[cxm] : u_xL   * hbar[c];
    float qfx_S = (u_self >= 0.f) ? u_self * hbar[c]   : u_self * hbar[cxp];
    float qfx_R = (u_xR   >= 0.f) ? u_xR   * hbar[cxp] : u_xR   * hbar[cxpp];

    float qBarX_L  = 0.5f * (qfx_L + qfx_S);             // at cell (i,   j)
    float qBarX_R  = 0.5f * (qfx_S + qfx_R);             // at cell (i+1, j)
    float uStarX_L = (qBarX_L >= 0.f) ? u_xL   : u_self; // upwind ubarx at cell (i,j)
    float uStarX_R = (qBarX_R >  0.f) ? u_self : u_xR;   // upwind ubarx at cell (i+1,j)

    // y-direction qbary values at the four corners around our momentum cell.
    float v_BL = ubary[cym];     // (i+0.5, j)
    float v_BR = ubary[cxp_ym];  // (i+1.5, j)
    float v_TL = ubary[c];       // (i+0.5, j+1)
    float v_TR = ubary[cxp];     // (i+1.5, j+1)

    float qfy_BL = (v_BL >= 0.f) ? v_BL * hbar[cym]    : v_BL * hbar[c];
    float qfy_BR = (v_BR >= 0.f) ? v_BR * hbar[cxp_ym] : v_BR * hbar[cxp];
    float qfy_TL = (v_TL >= 0.f) ? v_TL * hbar[c]      : v_TL * hbar[cyp];
    float qfy_TR = (v_TR >= 0.f) ? v_TR * hbar[cxp]    : v_TR * hbar[cxp_yp];

    float qBarY_B  = 0.5f * (qfy_BL + qfy_BR);           // at corner (i+1, j)
    float qBarY_T  = 0.5f * (qfy_TL + qfy_TR);           // at corner (i+1, j+1)
    float uStarY_B = (qBarY_B >= 0.f) ? u_yB   : u_self;
    float uStarY_T = (qBarY_T >  0.f) ? u_self : u_yT;

    float h_face = fmaxf(0.01f, hbar[c] + hbar[cxp]);
    float xFluxDiff = (qBarX_R * uStarX_R - qBarX_L * uStarX_L) / (float)GRIDCELLSIZE_2D;
    float yFluxDiff = (qBarY_T * uStarY_T - qBarY_B * uStarY_B) / (float)GRIDCELLSIZE_2D;
    float xQDiff    = (qBarX_R - qBarX_L) / (float)GRIDCELLSIZE_2D;
    float yQDiff    = (qBarY_T - qBarY_B) / (float)GRIDCELLSIZE_2D;

    float uu = (2.f / h_face) * ((xFluxDiff + yFluxDiff) - u_self * (xQDiff + yQDiff));

    float un = u_self - TIMESTEP_2D * uu;
    un += -GRAVITY * TIMESTEP_2D *
          ((terrain[cxp] + hbar[cxp]) - (terrain[c] + hbar[c])) / (float)GRIDCELLSIZE_2D;
    ubarxNew[c] = LimitVelocity_d(un);
}

// Symmetric counterpart for ubary (y-face), with x <-> y swapped.
__global__ void kernel_swe_momentum_y(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubary,
    const float* __restrict__ hbar,
    const float* __restrict__ terrain,
    float* __restrict__ ubaryNew)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;

    int im1 = clamp_x(i - 1);
    int ip1 = clamp_x(i + 1);
    int jm1 = clamp_y(j - 1);
    int jp1 = clamp_y(j + 1);
    int jp2 = clamp_y(j + 2);

    int c       = idx2(i,   j);
    int cym     = idx2(i,   jm1);
    int cyp     = idx2(i,   jp1);
    int cypp    = idx2(i,   jp2);
    int cxm     = idx2(im1, j);
    int cxp     = idx2(ip1, j);
    int cyp_xm  = idx2(im1, jp1);
    int cyp_xp  = idx2(ip1, jp1);

    float v_self = ubary[c];
    float v_yB   = ubary[cym];
    float v_yT   = ubary[cyp];
    float v_xL   = ubary[cxm];
    float v_xR   = ubary[cxp];

    // y-direction qbary at the three faces straddling our momentum cell.
    float qfy_B = (v_yB   >= 0.f) ? v_yB   * hbar[cym] : v_yB   * hbar[c];
    float qfy_S = (v_self >= 0.f) ? v_self * hbar[c]   : v_self * hbar[cyp];
    float qfy_T = (v_yT   >= 0.f) ? v_yT   * hbar[cyp] : v_yT   * hbar[cypp];

    float qBarY_B  = 0.5f * (qfy_B + qfy_S);             // at cell (i, j)
    float qBarY_T  = 0.5f * (qfy_S + qfy_T);             // at cell (i, j+1)
    float vStarY_B = (qBarY_B >= 0.f) ? v_yB   : v_self;
    float vStarY_T = (qBarY_T >  0.f) ? v_self : v_yT;

    // x-direction qbarx at the four corners around our momentum cell.
    float u_LB = ubarx[cxm];     // (i,   j+0.5)
    float u_RB = ubarx[c];       // (i+1, j+0.5)
    float u_LT = ubarx[cyp_xm];  // (i,   j+1.5)
    float u_RT = ubarx[cyp];     // (i+1, j+1.5)

    float qfx_LB = (u_LB >= 0.f) ? u_LB * hbar[cxm]    : u_LB * hbar[c];
    float qfx_RB = (u_RB >= 0.f) ? u_RB * hbar[c]      : u_RB * hbar[cxp];
    float qfx_LT = (u_LT >= 0.f) ? u_LT * hbar[cyp_xm] : u_LT * hbar[cyp];
    float qfx_RT = (u_RT >= 0.f) ? u_RT * hbar[cyp]    : u_RT * hbar[cyp_xp];

    float qBarX_L  = 0.5f * (qfx_LB + qfx_LT);           // at corner (i,   j+1)
    float qBarX_R  = 0.5f * (qfx_RB + qfx_RT);           // at corner (i+1, j+1)
    float vStarX_L = (qBarX_L >= 0.f) ? v_xL   : v_self;
    float vStarX_R = (qBarX_R >  0.f) ? v_self : v_xR;

    float h_face = fmaxf(0.01f, hbar[c] + hbar[cyp]);
    float yFluxDiff = (qBarY_T * vStarY_T - qBarY_B * vStarY_B) / (float)GRIDCELLSIZE_2D;
    float xFluxDiff = (qBarX_R * vStarX_R - qBarX_L * vStarX_L) / (float)GRIDCELLSIZE_2D;
    float yQDiff    = (qBarY_T - qBarY_B) / (float)GRIDCELLSIZE_2D;
    float xQDiff    = (qBarX_R - qBarX_L) / (float)GRIDCELLSIZE_2D;

    float vv = (2.f / h_face) * ((yFluxDiff + xFluxDiff) - v_self * (yQDiff + xQDiff));

    float vn = v_self - TIMESTEP_2D * vv;
    vn += -GRAVITY * TIMESTEP_2D *
          ((terrain[cyp] + hbar[cyp]) - (terrain[c] + hbar[c])) / (float)GRIDCELLSIZE_2D;
    ubaryNew[c] = LimitVelocity_d(vn);
}

__global__ void kernel_ubar_to_qbar_x(
    const float* __restrict__ ubarxNew,
    const float* __restrict__ hbar,
    float* __restrict__ qbarx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);

    float u = ubarxNew[c];
    qbarx[c] = (u >= 0.f) ? u * hbar[c] : u * hbar[cxp];
}

__global__ void kernel_ubar_to_qbar_y(
    const float* __restrict__ ubaryNew,
    const float* __restrict__ hbar,
    float* __restrict__ qbary)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cyp = idx2(i, clamp_y(j + 1));

    float v = ubaryNew[c];
    qbary[c] = (v >= 0.f) ? v * hbar[c] : v * hbar[cyp];
}

// ============================================================================
// 4.4  Transport surface flow rate / surface height through bulk velocity
//      (paper §4.4, Algorithms 3 & 4)
// ============================================================================

// Semi-Lagrangian advection of qtildex along the 2D bulk flow, sampled with
// bicubic Catmull-Rom and clamped to the local 2x2 inner block.
__global__ void kernel_advect_qtildex(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubary,
    const float* __restrict__ ubaryNew,
    const float* __restrict__ h,
    const float* __restrict__ qtildex_in,
    float* __restrict__ qtildex_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cxp     = idx2(clamp_x(i + 1), j);
    int cym     = idx2(i, clamp_y(j - 1));
    int cxp_ym  = idx2(clamp_x(i + 1), clamp_y(j - 1));

    float u_face = 0.5f * (ubarx[c] + ubarxNew[c]);
    // v at the x-face is averaged from the four nearest y-faces, each itself
    // averaged in time to the timestep midpoint.
    float v_face = 0.25f * (
          0.5f * (ubary[cym]    + ubaryNew[cym])
        + 0.5f * (ubary[cxp_ym] + ubaryNew[cxp_ym])
        + 0.5f * (ubary[c]      + ubaryNew[c])
        + 0.5f * (ubary[cxp]    + ubaryNew[cxp]));

    float sx  = (float)i - TIMESTEP_2D * u_face;
    float sy  = (float)j - TIMESTEP_2D * v_face;
    float val = SampleCubicClamped2D_d(sx, sy, qtildex_in);

    if ((u_face >= 0.f && h[c]   < 0.01f) ||
        (u_face <  0.f && h[cxp] < 0.01f))
        val = 0.f;
    qtildex_out[c] = val;
}

__global__ void kernel_advect_qtildey(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubary,
    const float* __restrict__ ubaryNew,
    const float* __restrict__ h,
    const float* __restrict__ qtildey_in,
    float* __restrict__ qtildey_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cyp     = idx2(i, clamp_y(j + 1));
    int cxm     = idx2(clamp_x(i - 1), j);
    int cyp_xm  = idx2(clamp_x(i - 1), clamp_y(j + 1));

    float v_face = 0.5f * (ubary[c] + ubaryNew[c]);
    float u_face = 0.25f * (
          0.5f * (ubarx[cxm]    + ubarxNew[cxm])
        + 0.5f * (ubarx[c]      + ubarxNew[c])
        + 0.5f * (ubarx[cyp_xm] + ubarxNew[cyp_xm])
        + 0.5f * (ubarx[cyp]    + ubarxNew[cyp]));

    float sx  = (float)i - TIMESTEP_2D * u_face;
    float sy  = (float)j - TIMESTEP_2D * v_face;
    float val = SampleCubicClamped2D_d(sx, sy, qtildey_in);

    if ((v_face >= 0.f && h[c]   < 0.01f) ||
        (v_face <  0.f && h[cyp] < 0.01f))
        val = 0.f;
    qtildey_out[c] = val;
}

// G = min(-div u, -gamma * div u),  qtildex *= exp(G * dt)
// Divergence is evaluated at the qtildex face using a 2-cell-wide central
// difference in x and the y-face averaged divergence from the four surrounding
// y-faces.
__global__ void kernel_div_ubar_qtildex(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubary,
    const float* __restrict__ ubaryNew,
    float* __restrict__ qtildex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cxm     = idx2(clamp_x(i - 1), j);
    int cxp     = idx2(clamp_x(i + 1), j);
    int cym     = idx2(i, clamp_y(j - 1));
    int cxp_ym  = idx2(clamp_x(i + 1), clamp_y(j - 1));

    float ux_m1 = 0.5f * (ubarxNew[cxm] + ubarx[cxm]);
    float ux_p1 = 0.5f * (ubarxNew[cxp] + ubarx[cxp]);
    float dux_dx = (ux_p1 - ux_m1) / (2.f * (float)GRIDCELLSIZE_2D);

    float uy_top = 0.5f * (
          0.5f * (ubaryNew[c]   + ubary[c])
        + 0.5f * (ubaryNew[cxp] + ubary[cxp]));
    float uy_bot = 0.5f * (
          0.5f * (ubaryNew[cym]    + ubary[cym])
        + 0.5f * (ubaryNew[cxp_ym] + ubary[cxp_ym]));
    float duy_dy = (uy_top - uy_bot) / (float)GRIDCELLSIZE_2D;

    float div_ubar = dux_dx + duy_dy;
    if (div_ubar < 0.f) div_ubar *= 0.25f;       // gamma = 1/4 amplification damping
    qtildex[c] *= expf(-div_ubar * TIMESTEP_2D);
}

__global__ void kernel_div_ubar_qtildey(
    const float* __restrict__ ubarx,
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubary,
    const float* __restrict__ ubaryNew,
    float* __restrict__ qtildey)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cxm     = idx2(clamp_x(i - 1), j);
    int cxp     = idx2(clamp_x(i + 1), j);
    int cym     = idx2(i, clamp_y(j - 1));
    int cyp     = idx2(i, clamp_y(j + 1));
    int cxm_yp  = idx2(clamp_x(i - 1), clamp_y(j + 1));

    float uy_m1 = 0.5f * (ubaryNew[cym] + ubary[cym]);
    float uy_p1 = 0.5f * (ubaryNew[cyp] + ubary[cyp]);
    float duy_dy = (uy_p1 - uy_m1) / (2.f * (float)GRIDCELLSIZE_2D);

    float ux_right = 0.5f * (
          0.5f * (ubarxNew[c]   + ubarx[c])
        + 0.5f * (ubarxNew[cyp] + ubarx[cyp]));
    float ux_left  = 0.5f * (
          0.5f * (ubarxNew[cxm]    + ubarx[cxm])
        + 0.5f * (ubarxNew[cxm_yp] + ubarx[cxm_yp]));
    float dux_dx = (ux_right - ux_left) / (float)GRIDCELLSIZE_2D;

    float div_ubar = dux_dx + duy_dy;
    if (div_ubar < 0.f) div_ubar *= 0.25f;
    qtildey[c] *= expf(-div_ubar * TIMESTEP_2D);
}

__global__ void kernel_div_ubar_htilde(
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubaryNew,
    float* __restrict__ htilde)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxm = idx2(clamp_x(i - 1), j);
    int cym = idx2(i, clamp_y(j - 1));

    float dux_dx = (ubarxNew[c] - ubarxNew[cxm]) / (float)GRIDCELLSIZE_2D;
    float duy_dy = (ubaryNew[c] - ubaryNew[cym]) / (float)GRIDCELLSIZE_2D;
    float div_ubar = dux_dx + duy_dy;
    if (div_ubar < 0.f) div_ubar *= 0.25f;
    htilde[c] *= expf(-div_ubar * TIMESTEP_2D);
}

// ============================================================================
// 4.5  Bulk-advected surface displacement -> flux update of h (paper §4.5)
// ============================================================================

__global__ void kernel_compute_advectHFRx(
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubaryNew,
    const float* __restrict__ htilde,
    float* __restrict__ advectHFRx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cxp     = idx2(clamp_x(i + 1), j);
    int cym     = idx2(i, clamp_y(j - 1));
    int cxp_ym  = idx2(clamp_x(i + 1), clamp_y(j - 1));

    float u = ubarxNew[c];
    float v = 0.25f * (ubaryNew[c] + ubaryNew[cxp]
                     + ubaryNew[cym] + ubaryNew[cxp_ym]);

    // World position of x-face is (i+1, j+0.5); htilde array index = world - (0.5, 0.5).
    float sx = (float)i + 0.5f - 0.5f * TIMESTEP_2D * u;
    float sy = (float)j        - 0.5f * TIMESTEP_2D * v;
    advectHFRx[c] = u * SampleCubicClamped2D_d(sx, sy, htilde);
}

__global__ void kernel_compute_advectHFRy(
    const float* __restrict__ ubarxNew,
    const float* __restrict__ ubaryNew,
    const float* __restrict__ htilde,
    float* __restrict__ advectHFRy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c       = idx2(i, j);
    int cyp     = idx2(i, clamp_y(j + 1));
    int cxm     = idx2(clamp_x(i - 1), j);
    int cxm_yp  = idx2(clamp_x(i - 1), clamp_y(j + 1));

    float v = ubaryNew[c];
    float u = 0.25f * (ubarxNew[c] + ubarxNew[cyp]
                     + ubarxNew[cxm] + ubarxNew[cxm_yp]);

    float sx = (float)i        - 0.5f * TIMESTEP_2D * u;
    float sy = (float)j + 0.5f - 0.5f * TIMESTEP_2D * v;
    advectHFRy[c] = v * SampleCubicClamped2D_d(sx, sy, htilde);
}

__global__ void kernel_h_update_from_advectHFR(
    const float* __restrict__ advectHFRx,
    const float* __restrict__ advectHFRy,
    const float* __restrict__ h_in,
    const float* __restrict__ terrain,
    float* __restrict__ h_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int im1 = clamp_x(i - 1);
    int ip1 = clamp_x(i + 1);
    int jm1 = clamp_y(j - 1);
    int jp1 = clamp_y(j + 1);
    int c   = idx2(i,   j);
    int cxm = idx2(im1, j);
    int cxp = idx2(ip1, j);
    int cym = idx2(i,   jm1);
    int cyp = idx2(i,   jp1);

    float qL = Limit_flow_rate_d(advectHFRx[cxm], h_in[cxm], h_in[c]);
    float qR = Limit_flow_rate_d(advectHFRx[c],   h_in[c],   h_in[cxp]);
    float qB = Limit_flow_rate_d(advectHFRy[cym], h_in[cym], h_in[c]);
    float qT = Limit_flow_rate_d(advectHFRy[c],   h_in[c],   h_in[cyp]);

    if (((h_in[cxm] == 0.f) && (h_in[c]  == 0.f))
        || StopFlowOnTerrainBoundary_x_d(im1, j, h_in, terrain))
        qL = 0.f;
    if (((h_in[c]  == 0.f) && (h_in[cxp] == 0.f))
        || StopFlowOnTerrainBoundary_x_d(i,   j, h_in, terrain))
        qR = 0.f;
    if (((h_in[cym] == 0.f) && (h_in[c]  == 0.f))
        || StopFlowOnTerrainBoundary_y_d(i, jm1, h_in, terrain))
        qB = 0.f;
    if (((h_in[c]  == 0.f) && (h_in[cyp] == 0.f))
        || StopFlowOnTerrainBoundary_y_d(i, j,   h_in, terrain))
        qT = 0.f;

    h_out[c] = fmaxf(0.f, h_in[c]
                  - TIMESTEP_2D / (float)GRIDCELLSIZE_2D * (qR - qL)
                  - TIMESTEP_2D / (float)GRIDCELLSIZE_2D * (qT - qB));
}

// ============================================================================
// 4.6  Recombine flow rates and final height integration (Eq. 16 / 17)
// ============================================================================

__global__ void kernel_recombine_qx(
    const float* __restrict__ qbarx,
    const float* __restrict__ qtildex,
    const float* __restrict__ h,
    const float* __restrict__ terrain,
    float* __restrict__ qx,
    int boundaryMode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);

    float q = Limit_flow_rate_d(qbarx[c] + qtildex[c], h[c], h[cxp]);
    bool stop = StopFlowOnTerrainBoundary_x_d(i, j, h, terrain);
    // The right-most x-face wraps to cell (N-1) itself due to clamp_x; it is a
    // degenerate face and must always be zero so it does not contribute to
    // the divergence in kernel_height_integration.
    bool degenerateFace   = (i >= GRIDRESOLUTION_2D - 1);
    bool reflectiveWall_x = (boundaryMode == 0)
                          && ((i == 0) || (i >= GRIDRESOLUTION_2D - 2));
    if (stop || degenerateFace || reflectiveWall_x) q = 0.f;
    qx[c] = q;
}

__global__ void kernel_recombine_qy(
    const float* __restrict__ qbary,
    const float* __restrict__ qtildey,
    const float* __restrict__ h,
    const float* __restrict__ terrain,
    float* __restrict__ qy,
    int boundaryMode)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cyp = idx2(i, clamp_y(j + 1));

    float q = Limit_flow_rate_d(qbary[c] + qtildey[c], h[c], h[cyp]);
    bool stop = StopFlowOnTerrainBoundary_y_d(i, j, h, terrain);
    bool degenerateFace   = (j >= GRIDRESOLUTION_2D - 1);
    bool reflectiveWall_y = (boundaryMode == 0)
                          && ((j == 0) || (j >= GRIDRESOLUTION_2D - 2));
    if (stop || degenerateFace || reflectiveWall_y) q = 0.f;
    qy[c] = q;
}

// ----------------------------------------------------------------------------
// Sponge layer (absorbing boundary).  In a band of `spongeWidth` cells along
// each domain edge, we apply a smoothly ramped per-step damping factor s(i,j)
// to all perturbation quantities and nudge h / hbar toward the rest water
// level.  s = 0 in the interior (band width away from any boundary) and grows
// quadratically up to ~spongeStrength right at the edge.  The outermost cells
// thus rapidly relax toward a quiescent constant-h state, behaving as an
// absorbing boundary that lets outgoing waves dissipate.
// ----------------------------------------------------------------------------
__global__ void kernel_apply_sponge(
    float* __restrict__ h,
    float* __restrict__ hbar,
    float* __restrict__ htilde,
    float* __restrict__ qx,
    float* __restrict__ qy,
    float* __restrict__ qbarx,
    float* __restrict__ qbary,
    float* __restrict__ qtildex,
    float* __restrict__ qtildey,
    const float* __restrict__ terrain,
    int   spongeWidth,
    float spongeStrength,
    float restWaterLevel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    if (spongeWidth <= 0) return;

    int dxL = i;
    int dxR = GRIDRESOLUTION_2D - 1 - i;
    int dyB = j;
    int dyT = GRIDRESOLUTION_2D - 1 - j;
    int d   = dxL < dxR ? dxL : dxR;
    int dy  = dyB < dyT ? dyB : dyT;
    if (dy < d) d = dy;
    if (d >= spongeWidth) return;

    float t = 1.f - (float)d / (float)spongeWidth;       // 0..1, 1 at edge
    float s = fminf(0.99f, t * t * spongeStrength);      // damping per step
    float keep = 1.f - s;

    int c = idx2(i, j);
    htilde [c] *= keep;
    qtildex[c] *= keep;
    qtildey[c] *= keep;
    qbarx  [c] *= keep;
    qbary  [c] *= keep;
    qx     [c] *= keep;
    qy     [c] *= keep;

    float target = fmaxf(0.f, restWaterLevel - terrain[c]);
    h   [c] += s * (target - h[c]);
    hbar[c] += s * (target - hbar[c]);
}

__global__ void kernel_height_integration(
    const float* __restrict__ qx,
    const float* __restrict__ qy,
    float* __restrict__ h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxm = idx2(clamp_x(i - 1), j);
    int cym = idx2(i, clamp_y(j - 1));

    float dqx_dx = (qx[c] - qx[cxm]) / (float)GRIDCELLSIZE_2D;
    float dqy_dy = (qy[c] - qy[cym]) / (float)GRIDCELLSIZE_2D;
    h[c] = fmaxf(0.f, h[c] + TIMESTEP_2D * -(dqx_dx + dqy_dy));
}

__global__ void kernel_qx_final_limit(
    const float* __restrict__ h,
    float* __restrict__ qx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cxp = idx2(clamp_x(i + 1), j);
    qx[c] = Limit_flow_rate_d(qx[c], h[c], h[cxp]);
}

__global__ void kernel_qy_final_limit(
    const float* __restrict__ h,
    float* __restrict__ qy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= GRIDRESOLUTION_2D || j >= GRIDRESOLUTION_2D) return;
    int c   = idx2(i, j);
    int cyp = idx2(i, clamp_y(j + 1));
    qy[c] = Limit_flow_rate_d(qy[c], h[c], h[cyp]);
}

// ============================================================================
// Sim2D member functions
// ============================================================================

Sim2D::Sim2D()
    : terrain_d(nullptr), h_d(nullptr), qx_d(nullptr), qy_d(nullptr),
      hbarOld_d(nullptr), htildeOld_d(nullptr),
      hbar_d(nullptr), qbarx_d(nullptr), qbary_d(nullptr),
      htilde_d(nullptr), qtildex_d(nullptr), qtildey_d(nullptr),
      alpha_x_d(nullptr), alpha_y_d(nullptr),
      alpha_qbarx_d(nullptr), alpha_qbary_d(nullptr),
      hbar_dummy_d(nullptr), qbarx_dummy_d(nullptr), qbary_dummy_d(nullptr),
      ubarx_d(nullptr), ubarxNew_d(nullptr),
      ubary_d(nullptr), ubaryNew_d(nullptr),
      qtildex_dummy_d(nullptr), qtildey_dummy_d(nullptr),
      advectHFRx_d(nullptr), advectHFRy_d(nullptr),
      h_dummy_d(nullptr),
      fwd_buf_d(nullptr), inv_buf_d(nullptr),
      plan_fwd(0), plan_inv(0), time(0.f)
{
    const size_t bytesF = (size_t)GRID_N2_2D * sizeof(float);

    CUDA_CHECK(cudaMalloc(&terrain_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&h_d,              bytesF));
    CUDA_CHECK(cudaMalloc(&qx_d,             bytesF));
    CUDA_CHECK(cudaMalloc(&qy_d,             bytesF));
    CUDA_CHECK(cudaMalloc(&hbarOld_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&htildeOld_d,      bytesF));
    CUDA_CHECK(cudaMalloc(&hbar_d,           bytesF));
    CUDA_CHECK(cudaMalloc(&qbarx_d,          bytesF));
    CUDA_CHECK(cudaMalloc(&qbary_d,          bytesF));
    CUDA_CHECK(cudaMalloc(&htilde_d,         bytesF));
    CUDA_CHECK(cudaMalloc(&qtildex_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&qtildey_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_x_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_y_d,        bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_qbarx_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&alpha_qbary_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&hbar_dummy_d,     bytesF));
    CUDA_CHECK(cudaMalloc(&qbarx_dummy_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&qbary_dummy_d,    bytesF));
    CUDA_CHECK(cudaMalloc(&ubarx_d,          bytesF));
    CUDA_CHECK(cudaMalloc(&ubarxNew_d,       bytesF));
    CUDA_CHECK(cudaMalloc(&ubary_d,          bytesF));
    CUDA_CHECK(cudaMalloc(&ubaryNew_d,       bytesF));
    CUDA_CHECK(cudaMalloc(&qtildex_dummy_d,  bytesF));
    CUDA_CHECK(cudaMalloc(&qtildey_dummy_d,  bytesF));
    CUDA_CHECK(cudaMalloc(&advectHFRx_d,     bytesF));
    CUDA_CHECK(cudaMalloc(&advectHFRy_d,     bytesF));
    CUDA_CHECK(cudaMalloc(&h_dummy_d,        bytesF));

    CUDA_CHECK(cudaMalloc(&fwd_buf_d, 3 * (size_t)GRID_N2_2D * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&inv_buf_d, 2 * (size_t)DEPTH_NUM_2D * (size_t)GRID_N2_2D * sizeof(cufftComplex)));

    int n[2] = { GRIDRESOLUTION_2D, GRIDRESOLUTION_2D };
    CUFFT_CHECK(cufftPlanMany(&plan_fwd, 2, n,
                              NULL, 1, GRID_N2_2D,
                              NULL, 1, GRID_N2_2D,
                              CUFFT_C2C, /*batch=*/ 3));
    CUFFT_CHECK(cufftPlanMany(&plan_inv, 2, n,
                              NULL, 1, GRID_N2_2D,
                              NULL, 1, GRID_N2_2D,
                              CUFFT_C2C, /*batch=*/ 2 * DEPTH_NUM_2D));

    CUDA_CHECK(cudaMemcpyToSymbol(c_Depth_2D, Depth_2D, DEPTH_NUM_2D * sizeof(float)));

    ResetTerrain(0);
    ResetWater(0, 12.0f);
}

int Sim2D::Release(void)
{
    if (plan_fwd) cufftDestroy(plan_fwd);
    if (plan_inv) cufftDestroy(plan_inv);
    plan_fwd = plan_inv = 0;

    cudaFree(terrain_d);        terrain_d        = nullptr;
    cudaFree(h_d);              h_d              = nullptr;
    cudaFree(qx_d);             qx_d             = nullptr;
    cudaFree(qy_d);             qy_d             = nullptr;
    cudaFree(hbarOld_d);        hbarOld_d        = nullptr;
    cudaFree(htildeOld_d);      htildeOld_d      = nullptr;
    cudaFree(hbar_d);           hbar_d           = nullptr;
    cudaFree(qbarx_d);          qbarx_d          = nullptr;
    cudaFree(qbary_d);          qbary_d          = nullptr;
    cudaFree(htilde_d);         htilde_d         = nullptr;
    cudaFree(qtildex_d);        qtildex_d        = nullptr;
    cudaFree(qtildey_d);        qtildey_d        = nullptr;
    cudaFree(alpha_x_d);        alpha_x_d        = nullptr;
    cudaFree(alpha_y_d);        alpha_y_d        = nullptr;
    cudaFree(alpha_qbarx_d);    alpha_qbarx_d    = nullptr;
    cudaFree(alpha_qbary_d);    alpha_qbary_d    = nullptr;
    cudaFree(hbar_dummy_d);     hbar_dummy_d     = nullptr;
    cudaFree(qbarx_dummy_d);    qbarx_dummy_d    = nullptr;
    cudaFree(qbary_dummy_d);    qbary_dummy_d    = nullptr;
    cudaFree(ubarx_d);          ubarx_d          = nullptr;
    cudaFree(ubarxNew_d);       ubarxNew_d       = nullptr;
    cudaFree(ubary_d);          ubary_d          = nullptr;
    cudaFree(ubaryNew_d);       ubaryNew_d       = nullptr;
    cudaFree(qtildex_dummy_d);  qtildex_dummy_d  = nullptr;
    cudaFree(qtildey_dummy_d);  qtildey_dummy_d  = nullptr;
    cudaFree(advectHFRx_d);     advectHFRx_d     = nullptr;
    cudaFree(advectHFRy_d);     advectHFRy_d     = nullptr;
    cudaFree(h_dummy_d);        h_dummy_d        = nullptr;
    cudaFree(fwd_buf_d);        fwd_buf_d        = nullptr;
    cudaFree(inv_buf_d);        inv_buf_d        = nullptr;
    return 0;
}

// ----------------------------------------------------------------------------
// Init functions  (host shadow construction + H2D upload)
// ----------------------------------------------------------------------------

// type: 0 = flat, 1 = 2D hills + central island
void Sim2D::ResetTerrain(int type)
{
    const float ABSSHIFT = fabsf((float)TERRAIN_HEIGHT_SHIFT_INIT_2D);

    for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            int c = j * GRIDRESOLUTION_2D + i;
            if (type == 0) {
                terrain[c] = -ABSSHIFT;
            } else {
                float u = (float)i / (float)GRIDRESOLUTION_2D;
                float v = (float)j / (float)GRIDRESOLUTION_2D;
                float hill = (-1.f + 0.1f
                              + 0.05f * (u + v)
                              + 0.02f * sinf(20.f * u)
                              + 0.02f * sinf(20.f * v)
                              + 0.45f * sinf(2.5f * u)
                              + 0.45f * sinf(2.5f * v));
                float du = u - 0.5f;
                float dv = v - 0.5f;
                float r2 = du * du + dv * dv;
                hill += 0.4f * expf(-r2 * 30.f);   // central island
                terrain[c] = hill * ABSSHIFT;
            }
        }
    }

    // Reflective walls on all 4 borders (high terrain).  Skipped in
    // absorbing mode so waves can travel into the sponge band and dissipate.
    if (boundaryMode == 0) {
        const float WALL = 1.8f * ABSSHIFT;
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            terrain[i]                                                     = WALL;
            terrain[(GRIDRESOLUTION_2D - 1) * GRIDRESOLUTION_2D + i]       = WALL;
        }
        for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
            terrain[j * GRIDRESOLUTION_2D + 0]                             = WALL;
            terrain[j * GRIDRESOLUTION_2D + GRIDRESOLUTION_2D - 1]         = WALL;
        }
    }

    CUDA_CHECK(cudaMemcpy(terrain_d, terrain,
                          GRID_N2_2D * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// type: 0 = constant level, 1 = dam break (left x-half empty),
//       2 = sloped along x, 3 = central cosine droplet
void Sim2D::ResetWater(int type, float level)
{
    const float SHIFT = (float)TERRAIN_HEIGHT_SHIFT_INIT_2D;

    for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            int c = j * GRIDRESOLUTION_2D + i;
            if (type == 0) {
                h[c] = fmaxf(0.f, level - terrain[c]);
            } else if (type == 1) {
                if (i <= GRIDRESOLUTION_2D / 2) h[c] = 0.f;
                else                            h[c] = fmaxf(0.f, level - terrain[c]);
            } else if (type == 2) {
                float t = -0.5f + (float)i / (float)GRIDRESOLUTION_2D;
                h[c] = fmaxf(0.f, level
                            + (2.f * t * fabsf(t)) * fabsf(0.5f * SHIFT)
                            - terrain[c]);
            } else {
                float du = (float)i - 0.5f * (float)GRIDRESOLUTION_2D;
                float dv = (float)j - 0.5f * (float)GRIDRESOLUTION_2D;
                float r  = sqrtf(du * du + dv * dv);
                float bump = 0.f;
                const float R = 20.f;
                if (r < R) bump = 0.5f * (1.f + cosf(PI * r / R));
                h[c] = fmaxf(0.f, level + bump - terrain[c]);
            }

            hbar   [c] = h[c];
            qbarx  [c] = 0.f;
            qbary  [c] = 0.f;
            htilde [c] = 0.f;
            qtildex[c] = 0.f;
            qtildey[c] = 0.f;
            qx     [c] = 0.f;
            qy     [c] = 0.f;
        }
    }

    // Zero water on the four border rings to enforce the wall (reflective
    // mode only).  The sponge layer takes care of the boundary in absorbing
    // mode and needs the rest level present at the edge.
    if (boundaryMode == 0) {
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            h   [i] = 0.f;
            hbar[i] = 0.f;
            h   [(GRIDRESOLUTION_2D - 1) * GRIDRESOLUTION_2D + i] = 0.f;
            hbar[(GRIDRESOLUTION_2D - 1) * GRIDRESOLUTION_2D + i] = 0.f;
        }
        for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
            h   [j * GRIDRESOLUTION_2D + 0]                       = 0.f;
            hbar[j * GRIDRESOLUTION_2D + 0]                       = 0.f;
            h   [j * GRIDRESOLUTION_2D + GRIDRESOLUTION_2D - 1]   = 0.f;
            hbar[j * GRIDRESOLUTION_2D + GRIDRESOLUTION_2D - 1]   = 0.f;
        }
    }
    restWaterLevel = level;

    const size_t bytesF = (size_t)GRID_N2_2D * sizeof(float);
    CUDA_CHECK(cudaMemcpy(h_d,         h,    bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hbar_d,      hbar, bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(hbarOld_d,   hbar, bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(htildeOld_d, 0,    bytesF));
    CUDA_CHECK(cudaMemset(htilde_d,    0,    bytesF));
    CUDA_CHECK(cudaMemset(qbarx_d,     0,    bytesF));
    CUDA_CHECK(cudaMemset(qbary_d,     0,    bytesF));
    CUDA_CHECK(cudaMemset(qx_d,        0,    bytesF));
    CUDA_CHECK(cudaMemset(qy_d,        0,    bytesF));
    CUDA_CHECK(cudaMemset(qtildex_d,   0,    bytesF));
    CUDA_CHECK(cudaMemset(qtildey_d,   0,    bytesF));

    time = 0.f;
}

// xN, yN, size in (0..1); factor controls per-call magnitude.
void Sim2D::EditWaterLocal(float xN, float yN, float size, float factor)
{
    CUDA_CHECK(cudaMemcpy(h, h_d,
                          GRID_N2_2D * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            int c = j * GRIDRESOLUTION_2D + i;
            float du = (float)i / (float)GRIDRESOLUTION_2D - xN;
            float dv = (float)j / (float)GRIDRESOLUTION_2D - yN;
            float r  = sqrtf(du * du + dv * dv);
            if (r < size) {
                h[c] = fmaxf(0.f, h[c] + factor * 1.f);
            }
        }
    }
    // Re-enforce the wall on all 4 borders (reflective mode only).
    if (boundaryMode == 0) {
        for (int i = 0; i < GRIDRESOLUTION_2D; i++) {
            h[i]                                                   = 0.f;
            h[(GRIDRESOLUTION_2D - 1) * GRIDRESOLUTION_2D + i]     = 0.f;
        }
        for (int j = 0; j < GRIDRESOLUTION_2D; j++) {
            h[j * GRIDRESOLUTION_2D + 0]                           = 0.f;
            h[j * GRIDRESOLUTION_2D + GRIDRESOLUTION_2D - 1]       = 0.f;
        }
    }
    CUDA_CHECK(cudaMemcpy(h_d, h,
                          GRID_N2_2D * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void Sim2D::SyncToHost()
{
    const size_t bytesF = (size_t)GRID_N2_2D * sizeof(float);
    CUDA_CHECK(cudaMemcpy(terrain, terrain_d, bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h,       h_d,       bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qx,      qx_d,      bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qy,      qy_d,      bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hbar,    hbar_d,    bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qbarx,   qbarx_d,   bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qbary,   qbary_d,   bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(htilde,  htilde_d,  bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qtildex, qtildex_d, bytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qtildey, qtildey_d, bytesF, cudaMemcpyDeviceToHost));
}

// ============================================================================
// SimStep -- mirrors paper Algorithm 1 in the same block order as the 1D
// implementation.  Each block below corresponds to a comment-delimited section
// of Sim.cu.
// ============================================================================
void Sim2D::SimStep(bool SWEonly)
{
    const size_t bytesF = (size_t)GRID_N2_2D * sizeof(float);

    // ------------------------------------------------------------------
    // 4.1 Bulk vs. surface decomposition
    // ------------------------------------------------------------------
    kernel_init_decomp        <<<kGrid, kBlock>>>(terrain_d, h_d, qx_d, qy_d,
                                                  hbar_d, qbarx_d, qbary_d,
                                                  alpha_x_d, alpha_y_d);
    CUDA_CHECK_LAST();
    kernel_compute_alpha_qbar <<<kGrid, kBlock>>>(alpha_x_d, alpha_y_d,
                                                  alpha_qbarx_d, alpha_qbary_d);
    CUDA_CHECK_LAST();

    if (!SWEonly) {
        // Seed all three dummy buffers (the diffusion kernel only updates the
        // interior; the last row/column simply copies its input to the output).
        CUDA_CHECK(cudaMemcpyAsync(hbar_dummy_d,  hbar_d,  bytesF, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(qbarx_dummy_d, qbarx_d, bytesF, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpyAsync(qbary_dummy_d, qbary_d, bytesF, cudaMemcpyDeviceToDevice));

        float* h_in  = hbar_d;       float* h_out = hbar_dummy_d;
        float* qxin  = qbarx_d;      float* qxout = qbarx_dummy_d;
        float* qyin  = qbary_d;      float* qyout = qbary_dummy_d;
        // 128 iterations at K=0.24 keeps the same total diffusion as the 1D
        // implementation's 64 iterations at K=0.48 while staying within the 2D
        // explicit FTCS upper bound (0.25 = 0.5 / D, D = 2).
        for (int it = 0; it < 128; it++) {
            kernel_diffusion_step<<<kGrid, kBlock>>>(h_in, qxin, qyin,
                                                     h_out, qxout, qyout,
                                                     terrain_d,
                                                     alpha_x_d, alpha_y_d,
                                                     alpha_qbarx_d, alpha_qbary_d);
            std::swap(h_in,  h_out);
            std::swap(qxin,  qxout);
            std::swap(qyin,  qyout);
        }
        // After 128 swaps the data ends up in the originals, but stay defensive.
        if (h_in != hbar_d) {
            CUDA_CHECK(cudaMemcpyAsync(hbar_d,  h_in, bytesF, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(qbarx_d, qxin, bytesF, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpyAsync(qbary_d, qyin, bytesF, cudaMemcpyDeviceToDevice));
        }
    }

    kernel_finalize_decomposition<<<kGrid, kBlock>>>(terrain_d, h_d, qx_d, qy_d,
                                                     hbar_d, htilde_d,
                                                     qbarx_d, qbary_d,
                                                     qtildex_d, qtildey_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // 4.2 Surface velocity update via eWave (Algorithm 2)
    // ------------------------------------------------------------------
    kernel_pack_fwd_input<<<kGrid, kBlock>>>(htilde_d, htildeOld_d,
                                             qtildex_d, qtildey_d, fwd_buf_d);
    CUDA_CHECK_LAST();

    CUFFT_CHECK(cufftExecC2C(plan_fwd, fwd_buf_d, fwd_buf_d, CUFFT_FORWARD));

    kernel_freq_domain_2d<<<kGrid, kBlock>>>(fwd_buf_d, inv_buf_d);
    CUDA_CHECK_LAST();

    CUFFT_CHECK(cufftExecC2C(plan_inv, inv_buf_d, inv_buf_d, CUFFT_INVERSE));

    kernel_depth_interp_qtildex<<<kGrid, kBlock>>>(inv_buf_d, hbar_d, qtildex_d);
    CUDA_CHECK_LAST();
    kernel_depth_interp_qtildey<<<kGrid, kBlock>>>(inv_buf_d, hbar_d, qtildey_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // 4.3 SWE bulk simulation [Stelling03] -- full 2D
    // ------------------------------------------------------------------
    kernel_qbar_to_ubar_x<<<kGrid, kBlock>>>(qbarx_d, hbarOld_d, ubarx_d);
    CUDA_CHECK_LAST();
    kernel_qbar_to_ubar_y<<<kGrid, kBlock>>>(qbary_d, hbarOld_d, ubary_d);
    CUDA_CHECK_LAST();

    // store current hbar for next timestep
    CUDA_CHECK(cudaMemcpyAsync(hbarOld_d, hbar_d, bytesF, cudaMemcpyDeviceToDevice));

    kernel_swe_momentum_x<<<kGrid, kBlock>>>(ubarx_d, ubary_d, hbar_d, terrain_d, ubarxNew_d);
    CUDA_CHECK_LAST();
    kernel_swe_momentum_y<<<kGrid, kBlock>>>(ubarx_d, ubary_d, hbar_d, terrain_d, ubaryNew_d);
    CUDA_CHECK_LAST();

    kernel_ubar_to_qbar_x<<<kGrid, kBlock>>>(ubarxNew_d, hbar_d, qbarx_d);
    CUDA_CHECK_LAST();
    kernel_ubar_to_qbar_y<<<kGrid, kBlock>>>(ubaryNew_d, hbar_d, qbary_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // 4.4 advect qtildex/y + amplification damping for qtildex/y / htilde
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpyAsync(qtildex_dummy_d, qtildex_d, bytesF, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(qtildey_dummy_d, qtildey_d, bytesF, cudaMemcpyDeviceToDevice));
    kernel_advect_qtildex<<<kGrid, kBlock>>>(ubarx_d, ubarxNew_d, ubary_d, ubaryNew_d,
                                             h_d, qtildex_dummy_d, qtildex_d);
    CUDA_CHECK_LAST();
    kernel_advect_qtildey<<<kGrid, kBlock>>>(ubarx_d, ubarxNew_d, ubary_d, ubaryNew_d,
                                             h_d, qtildey_dummy_d, qtildey_d);
    CUDA_CHECK_LAST();
    kernel_div_ubar_qtildex<<<kGrid, kBlock>>>(ubarx_d, ubarxNew_d, ubary_d, ubaryNew_d, qtildex_d);
    CUDA_CHECK_LAST();
    kernel_div_ubar_qtildey<<<kGrid, kBlock>>>(ubarx_d, ubarxNew_d, ubary_d, ubaryNew_d, qtildey_d);
    CUDA_CHECK_LAST();
    kernel_div_ubar_htilde <<<kGrid, kBlock>>>(ubarxNew_d, ubaryNew_d, htilde_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // 4.5 bulk-advected surface displacement -> flux update of h
    // ------------------------------------------------------------------
    kernel_compute_advectHFRx<<<kGrid, kBlock>>>(ubarxNew_d, ubaryNew_d, htilde_d, advectHFRx_d);
    CUDA_CHECK_LAST();
    kernel_compute_advectHFRy<<<kGrid, kBlock>>>(ubarxNew_d, ubaryNew_d, htilde_d, advectHFRy_d);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(h_dummy_d, h_d, bytesF, cudaMemcpyDeviceToDevice));
    kernel_h_update_from_advectHFR<<<kGrid, kBlock>>>(advectHFRx_d, advectHFRy_d,
                                                     h_dummy_d, terrain_d, h_d);
    CUDA_CHECK_LAST();

    // ------------------------------------------------------------------
    // 4.6 recombine (Eq. 16) + height integration (Eq. 17) + final limit
    // ------------------------------------------------------------------
    kernel_recombine_qx<<<kGrid, kBlock>>>(qbarx_d, qtildex_d, h_d, terrain_d, qx_d, boundaryMode);
    CUDA_CHECK_LAST();
    kernel_recombine_qy<<<kGrid, kBlock>>>(qbary_d, qtildey_d, h_d, terrain_d, qy_d, boundaryMode);
    CUDA_CHECK_LAST();
    kernel_height_integration<<<kGrid, kBlock>>>(qx_d, qy_d, h_d);
    CUDA_CHECK_LAST();
    kernel_qx_final_limit<<<kGrid, kBlock>>>(h_d, qx_d);
    CUDA_CHECK_LAST();
    kernel_qy_final_limit<<<kGrid, kBlock>>>(h_d, qy_d);
    CUDA_CHECK_LAST();

    if (boundaryMode == 1) {
        kernel_apply_sponge<<<kGrid, kBlock>>>(
            h_d, hbar_d, htilde_d,
            qx_d, qy_d,
            qbarx_d, qbary_d,
            qtildex_d, qtildey_d,
            terrain_d,
            spongeWidth, spongeStrength, restWaterLevel);
        CUDA_CHECK_LAST();
    }

    time += TIMESTEP_2D;

    // Make the host shadow arrays usable to the renderer immediately after
    // every step.  Cost is 10 * N * N * sizeof(float) ~= 2.5 MB per step.
    SyncToHost();
}
