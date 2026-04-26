#pragma once

// CUDA / cuFFT 2D extension of the 1D dispersive SWE simulator.
// Mirrors the same algorithm as Sim.cu but on an N x N MAC (staggered) grid:
//
//   * Cell-centered  (i+0.5, j+0.5) : terrain, h, hbar, htilde, hbarOld, htildeOld
//   * x-face         (i+1,  j+0.5) : qx,  qbarx,  qtildex,  ubarx,  ubarxNew, advectHFRx
//   * y-face         (i+0.5, j+1)  : qy,  qbary,  qtildey,  ubary,  ubaryNew, advectHFRy
//
// All primary state lives on the GPU as device pointers (`*_d`). The same-named
// host arrays (terrain, h, qx, qy, hbar, qbarx, qbary, htilde, qtildex, qtildey)
// are "shadow" buffers populated by Sim2D::SyncToHost() and consumed by any
// renderer.

#include <cufft.h>

#ifndef GRAVITY
#define GRAVITY 9.80665f
#endif
#ifndef PI
#define PI 3.14159265359f
#endif

// 2D sim parameters
#define GRIDRESOLUTION_2D 256
#define GRIDCELLSIZE_2D 1            // do not change in this implementation
#define TIMESTEP_2D (1.f/60.f)
#define DEPTH_NUM_2D 4
const float Depth_2D[DEPTH_NUM_2D] = { 1.f, 4.f, 16.f, 64.f };
#define TERRAIN_HEIGHT_SHIFT_INIT_2D -10.f
#define TERRAIN_HEIGHT_SCALE_INIT_2D  20.f

// Total number of cells; used everywhere as the linearised size.
#define GRID_N2_2D (GRIDRESOLUTION_2D * GRIDRESOLUTION_2D)

class Sim2D
{
public:
    // ---------------------------------------------------------------------
    // Host shadow arrays (only valid after a call to SyncToHost()).
    // Index linearisation: idx(i, j) = j * GRIDRESOLUTION_2D + i
    // ---------------------------------------------------------------------
    float terrain [GRID_N2_2D];
    float h       [GRID_N2_2D];
    float qx      [GRID_N2_2D];
    float qy      [GRID_N2_2D];
    float hbar    [GRID_N2_2D];
    float qbarx   [GRID_N2_2D];
    float qbary   [GRID_N2_2D];
    float htilde  [GRID_N2_2D];
    float qtildex [GRID_N2_2D];
    float qtildey [GRID_N2_2D];

    // ---------------------------------------------------------------------
    // Device (GPU) state — primary simulation arrays.
    // ---------------------------------------------------------------------
    float* terrain_d;
    float* h_d;
    float* qx_d;
    float* qy_d;
    float* hbarOld_d;       // last-step bulk height (for leapfrog re-sample)
    float* htildeOld_d;     // last-step surface displacement (eWave time avg)
    float* hbar_d;
    float* qbarx_d;
    float* qbary_d;
    float* htilde_d;
    float* qtildex_d;
    float* qtildey_d;

    // Diffusion conductances
    float* alpha_x_d;       // x-face,  used for hbar x-bond
    float* alpha_y_d;       // y-face,  used for hbar y-bond
    float* alpha_qbarx_d;   // cell-center, used for qbarx x-bond
    float* alpha_qbary_d;   // cell-center, used for qbary y-bond

    // Ping-pong buffers for diffusion
    float* hbar_dummy_d;
    float* qbarx_dummy_d;
    float* qbary_dummy_d;

    // SWE bulk velocity (x-face / y-face)
    float* ubarx_d;
    float* ubarxNew_d;
    float* ubary_d;
    float* ubaryNew_d;

    // Surface advection scratch
    float* qtildex_dummy_d;
    float* qtildey_dummy_d;
    float* advectHFRx_d;
    float* advectHFRy_d;

    // h backup for divergence update
    float* h_dummy_d;

    // ---------------------------------------------------------------------
    // cuFFT state — single-precision complex (C2C), batched 2D.
    // fwd_buf_d : layout [ htildehat (N*N) | qtildexhat (N*N) | qtildeyhat (N*N) ], batch = 3
    // inv_buf_d : layout [ qtildex_depth0..3 (N*N each) | qtildey_depth0..3 (N*N each) ],
    //             batch = 2 * DEPTH_NUM_2D
    // ---------------------------------------------------------------------
    cufftComplex* fwd_buf_d;
    cufftComplex* inv_buf_d;
    cufftHandle   plan_fwd;
    cufftHandle   plan_inv;

    // time is exclusively used for video recording
    float time;

    // ---------------------------------------------------------------------
    // API
    // ---------------------------------------------------------------------
    Sim2D();
    int  Release(void);
    void ResetTerrain(int type);                                // 0=flat, 1=hills+island
    void ResetWater(int type, float level);                     // 0=const, 1=dam-x, 2=sloped, 3=cosine droplet
    void SimStep(bool SWEonly);                                 // advance by one timestep
    void EditWaterLocal(float xN, float yN, float size, float factor);

    // Copies device-side authoritative arrays into the host shadow buffers.
    // Call once per frame before reading any host-side member arrays.
    void SyncToHost();
};
