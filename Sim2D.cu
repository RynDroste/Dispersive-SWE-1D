// CUDA / cuFFT 2D extension of the 1D dispersive SWE simulator.

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


// Constant memory: the 4 sampled water depths used by the eWave Airy solver.
__constant__ float c_Depth_2D[DEPTH_NUM_2D];

// 2D launch geometry: 16x16 thread blocks tile the N x N domain.
static const dim3 kBlock(16, 16);
static const dim3 kGrid((GRIDRESOLUTION_2D + 15) / 16,
                       (GRIDRESOLUTION_2D + 15) / 16);

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