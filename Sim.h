#pragma once

// CUDA / cuFFT migration of the original alglib-based 1D simulator.
// All primary state lives on the GPU as device pointers (`*_d`).
// The same-named host arrays (terrain, h, q, hbar, qbar, htilde, qtilde) are
// "shadow" buffers that are populated by Sim::SyncToHost() and consumed by the
// renderer (WavePacketViewer).

#include <cufft.h>

#define GRAVITY 9.80665f
#define PI 3.14159265359f

// sim parameters
#define GRIDRESOLUTION 256
#define GRIDCELLSIZE 1      // this should not be changed in this implementation!
#define TIMESTEP (1.f/60.f)
#define DEPTH_NUM 4
const float Depth[DEPTH_NUM] = { 1.f, 4.f, 16.f, 64.f };
#define TERRAIN_HEIGHT_SHIFT_INIT -10.f
#define TERRAIN_HEIGHT_SCALE_INIT 20.f


class Sim
{
public:
    // ---------------------------------------------------------------------
    // Host shadow arrays (kept for renderer compatibility).
    // Only valid after a call to SyncToHost().
    // ---------------------------------------------------------------------
    float terrain[GRIDRESOLUTION];
    float h      [GRIDRESOLUTION];
    float q      [GRIDRESOLUTION];
    float hbar   [GRIDRESOLUTION];
    float qbar   [GRIDRESOLUTION];
    float htilde [GRIDRESOLUTION];
    float qtilde [GRIDRESOLUTION];

    // ---------------------------------------------------------------------
    // Device (GPU) state — primary simulation arrays.
    // ---------------------------------------------------------------------
    float* terrain_d;
    float* h_d;
    float* q_d;
    float* hbarOld_d;       // last-step bulk height (for leapfrog re-sample)
    float* htildeOld_d;     // last-step surface displacement (eWave time avg)
    float* hbar_d;
    float* qbar_d;
    float* htilde_d;
    float* qtilde_d;

    // Device-side scratch buffers
    float* alpha_hbar_d;
    float* alpha_qbar_d;
    float* hbar_dummy_d;    // ping-pong buffer for diffusion
    float* qbar_dummy_d;    // ping-pong buffer for diffusion
    float* ubar_d;
    float* ubarNew_d;
    float* qtilde_dummy_d;  // semi-Lagrangian source backup
    float* advectHFR_d;
    float* h_dummy_d;       // h backup for divergence update

    // ---------------------------------------------------------------------
    // cuFFT state — single-precision complex (C2C), batched 1D.
    // fwd_buf_d : layout [ htildehat (N) | qtildehat (N) ], batch = 2
    // inv_buf_d : layout [ depth0 (N) | depth1 (N) | ... ],  batch = DEPTH_NUM
    // ---------------------------------------------------------------------
    cufftComplex* fwd_buf_d;
    cufftComplex* inv_buf_d;
    cufftHandle   plan_fwd;
    cufftHandle   plan_inv;

    // time is exclusively used for video recording
    float time;

    // ---------------------------------------------------------------------
    // API (unchanged signatures, plus SyncToHost)
    // ---------------------------------------------------------------------
    Sim();
    int  Release(void);
    void ResetTerrain(int type);
    void ResetWater(int type, float level);
    void SimStep(bool SWEonly);                                // advect by one timestep
    void EditWaterLocal(float xCoord, float size, float factor);

    // Copies the device-side authoritative arrays (terrain, h, q, hbar,
    // qbar, htilde, qtilde) into the matching host shadow buffers above.
    // Call this once per frame before reading any host-side member arrays
    // (e.g. from the renderer).
    void SyncToHost();
};
