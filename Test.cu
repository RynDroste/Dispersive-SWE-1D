// ----------------------------------------------------------------------------
// Test.cu  ---  Standalone validation harness for the cuFFT/CUDA migration.
//
// Build (Windows / Visual Studio Developer PowerShell):
//   nvcc -O2 Test.cu Sim.cu -lcufft -o test_sim.exe
// Run:
//   .\test_sim.exe
//
// Exits with non-zero status if any test fails.
// ----------------------------------------------------------------------------

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>

#include "Sim.h"
#include "CudaCheck.h"

static int g_failures = 0;

#define EXPECT(cond, fmt, ...)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);                   \
            ++g_failures;                                                       \
        } else {                                                                \
            std::printf("  [ ok ] " fmt "\n", ##__VA_ARGS__);                   \
        }                                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// Test 1: cuFFT forward + inverse + 1/N normalization round-trips a known signal.
//         Verifies that the manual 1/N scaling we apply in
//         kernel_depth_interp_qtilde matches alglib's normalization convention.
// ----------------------------------------------------------------------------
static void TestFftRoundTrip()
{
    std::printf("[Test 1] cuFFT C2C round-trip (N = %d)\n", GRIDRESOLUTION);

    std::vector<cufftComplex> h_in(GRIDRESOLUTION);
    for (int i = 0; i < GRIDRESOLUTION; i++) {
        h_in[i].x = std::sinf(0.13f * i) + 0.5f * std::cosf(0.41f * i + 1.f);
        h_in[i].y = 0.f;
    }

    cufftComplex* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, GRIDRESOLUTION * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(d_buf, h_in.data(),
                          GRIDRESOLUTION * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, GRIDRESOLUTION, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftExecC2C(plan, d_buf, d_buf, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_buf, d_buf, CUFFT_INVERSE));
    cufftDestroy(plan);

    std::vector<cufftComplex> h_out(GRIDRESOLUTION);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_buf,
                          GRIDRESOLUTION * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_buf);

    float linf = 0.f;
    const float invN = 1.f / (float)GRIDRESOLUTION;
    for (int i = 0; i < GRIDRESOLUTION; i++) {
        float r = h_out[i].x * invN - h_in[i].x;
        float im = h_out[i].y * invN - h_in[i].y;
        linf = std::max(linf, std::fabs(r));
        linf = std::max(linf, std::fabs(im));
    }
    EXPECT(linf < 1e-4f, "L_inf round-trip error = %.3e (expect < 1e-4)", (double)linf);
}

// ----------------------------------------------------------------------------
// Test 2: Static water (no perturbation, no slope) must conserve total volume
//         exactly and not generate spurious motion.
// ----------------------------------------------------------------------------
static void TestStaticWaterConservation(int steps)
{
    std::printf("[Test 2] Static water mass conservation over %d steps\n", steps);

    Sim sim;
    sim.ResetTerrain(0);            // flat terrain
    sim.ResetWater(0, /*level=*/ 5.f); // calm water, surface at y = 5
    sim.SyncToHost();

    double m0 = 0.0;
    for (int x = 0; x < GRIDRESOLUTION; x++) m0 += (double)sim.h[x];

    for (int s = 0; s < steps; s++) sim.SimStep(/*SWEonly=*/ false);
    sim.SyncToHost();

    double m1 = 0.0;
    float maxAbsQ = 0.f;
    for (int x = 0; x < GRIDRESOLUTION; x++) {
        m1 += (double)sim.h[x];
        maxAbsQ = std::max(maxAbsQ, std::fabs(sim.q[x]));
    }
    double dm_rel = std::fabs(m1 - m0) / std::max(1e-12, std::fabs(m0));

    sim.Release();
    EXPECT(dm_rel < 1e-5,         "relative mass drift = %.3e (expect < 1e-5)", dm_rel);
    EXPECT(maxAbsQ < 1e-3f,       "max |q| in static run = %.3e (expect < 1e-3)", (double)maxAbsQ);
}

// ----------------------------------------------------------------------------
// Test 3: SWEonly mode produces volume-conserving non-trivial flow on a hill.
// ----------------------------------------------------------------------------
static void TestSWEOnlyBasicRun(int steps)
{
    std::printf("[Test 3] SWEonly run (hill terrain, sloped water) %d steps\n", steps);

    Sim sim;
    sim.ResetTerrain(1);                 // hill
    sim.ResetWater(2, /*level=*/ 0.f);   // sloped water
    sim.SyncToHost();

    double m0 = 0.0;
    for (int x = 0; x < GRIDRESOLUTION; x++) m0 += (double)sim.h[x];

    bool any_nan = false;
    for (int s = 0; s < steps; s++) {
        sim.SimStep(/*SWEonly=*/ true);
        if ((s % 30) == 0) {
            sim.SyncToHost();
            for (int x = 0; x < GRIDRESOLUTION; x++) {
                if (!std::isfinite(sim.h[x]) || !std::isfinite(sim.q[x])) {
                    any_nan = true; break;
                }
            }
        }
    }
    sim.SyncToHost();

    double m1 = 0.0;
    for (int x = 0; x < GRIDRESOLUTION; x++) m1 += (double)sim.h[x];
    double dm_rel = std::fabs(m1 - m0) / std::max(1e-12, std::fabs(m0));

    sim.Release();
    EXPECT(!any_nan,            "no NaN/Inf encountered during run");
    EXPECT(dm_rel < 1e-3,       "SWEonly mass drift = %.3e (expect < 1e-3 -- water hits walls)", dm_rel);
}

// ----------------------------------------------------------------------------
// Test 4: Standing wave dispersion -- excite a single Fourier mode with a known
//         wavelength on a constant-depth box, record the height at an antinode
//         for many cycles, and recover the period via autocorrelation.
//         Compares measured omega/k against the theoretical Airy phase speed
//         sqrt(g*tanh(k*h)/k). Reproduces a single point of paper Fig. 6.
//
//         The previous version used baseline = h(t=0), which is NOT the time
//         mean for an off-antinode probe; that yielded fake zero-crossings and
//         a phantom period.  This version:
//           1. Builds a custom h(x,0) = h_depth + amp*cos(2*pi*x/lambda) with
//              lambda chosen so an integer number of wavelengths fits the box.
//           2. Probes an antinode (h(probe,0) is an extremum).
//           3. Runs a short warmup, then records h[probe] for `trace_len` steps.
//           4. Subtracts the trace mean (the actual time-mean, not h(t=0)).
//           5. Computes autocorrelation R(tau) and picks tau* in [0.5*T, 2*T].
//           6. T_meas = tau* * dt;   ratio = T_meas / T_theoretical.
// ----------------------------------------------------------------------------
static void TestStandingWaveDispersion()
{
    const float h_depth  = 4.f;
    const float lambda   = 32.f;             // 8 full wavelengths fit in N=256
    const float amp      = 0.30f;            // initial perturbation amplitude
    const float k        = 2.f * PI / lambda;
    const float omega_th = std::sqrt(GRAVITY * k * std::tanh(k * h_depth));
    const float T_th     = 2.f * PI / omega_th;

    std::printf("[Test 4] Standing wave dispersion (lambda=%.1fm, h=%.1fm, T_th=%.3fs)\n",
                (double)lambda, (double)h_depth, (double)T_th);

    Sim sim;

    // Constant-depth scene with reflective walls at the boundaries.
    for (int x = 0; x < GRIDRESOLUTION; x++) sim.terrain[x] = -h_depth;
    sim.terrain[0]                  = 1.8f * 10.f;
    sim.terrain[GRIDRESOLUTION - 1] = 1.8f * 10.f;
    const size_t bytesF = (size_t)GRIDRESOLUTION * sizeof(float);
    CUDA_CHECK(cudaMemcpy(sim.terrain_d, sim.terrain, bytesF, cudaMemcpyHostToDevice));

    // Custom water initial condition: a single cosine mode of wavelength lambda.
    for (int x = 0; x < GRIDRESOLUTION; x++) {
        sim.h[x]      = h_depth + amp * std::cos(2.f * PI * (float)x / lambda);
        sim.hbar[x]   = sim.h[x];
        sim.qbar[x]   = 0.f;
        sim.htilde[x] = 0.f;
        sim.qtilde[x] = 0.f;
        sim.q[x]      = 0.f;
    }
    sim.h[0] = sim.h[GRIDRESOLUTION - 1] = 0.f;
    sim.hbar[0] = sim.hbar[GRIDRESOLUTION - 1] = 0.f;

    CUDA_CHECK(cudaMemcpy(sim.h_d,         sim.h,    bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.hbar_d,      sim.hbar, bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sim.hbarOld_d,   sim.hbar, bytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sim.htildeOld_d, 0, bytesF));
    CUDA_CHECK(cudaMemset(sim.htilde_d,    0, bytesF));
    CUDA_CHECK(cudaMemset(sim.qbar_d,      0, bytesF));
    CUDA_CHECK(cudaMemset(sim.q_d,         0, bytesF));
    CUDA_CHECK(cudaMemset(sim.qtilde_d,    0, bytesF));
    sim.time = 0.f;
    sim.SyncToHost();

    // Probe at an antinode: x = lambda is an extremum of cos(2*pi*x/lambda).
    const int probe = (int)lambda;
    if (probe < 1 || probe >= GRIDRESOLUTION - 1) {
        std::printf("  [skip] invalid probe index %d\n", probe);
        sim.Release();
        return;
    }

    // Warmup: let the box settle into the standing-wave state.
    const int warmup = 60;
    for (int s = 0; s < warmup; s++) sim.SimStep(/*SWEonly=*/ false);

    // Record the trace at the probe.
    const int trace_len = 2048;
    std::vector<float> trace(trace_len);
    for (int s = 0; s < trace_len; s++) {
        sim.SimStep(/*SWEonly=*/ false);
        sim.SyncToHost();
        trace[s] = sim.h[probe];
    }

    // Subtract trace mean.
    double sum = 0.0;
    for (float v : trace) sum += (double)v;
    const float mean = (float)(sum / (double)trace_len);
    for (float& v : trace) v -= mean;

    // Trace amplitude (rms and peak) -- if too small, the test is meaningless.
    double sum_sq = 0.0;
    float  peak   = 0.f;
    for (float v : trace) { sum_sq += (double)v * v; peak = std::max(peak, std::fabs(v)); }
    const float rms = (float)std::sqrt(sum_sq / (double)trace_len);

    // Autocorrelation: search the lag tau in [0.5 * T_th, 2.0 * T_th] (in steps)
    int tau_min = std::max(1, (int)(0.5f * T_th / (float)TIMESTEP));
    int tau_max = std::min(trace_len / 2 - 1, (int)(2.0f * T_th / (float)TIMESTEP));

    int    tau_best = tau_min;
    double R_best   = -1e300;
    for (int tau = tau_min; tau <= tau_max; tau++) {
        double R = 0.0;
        const int N = trace_len - tau;
        for (int t = 0; t < N; t++) R += (double)trace[t] * (double)trace[t + tau];
        if (R > R_best) { R_best = R; tau_best = tau; }
    }

    const float T_meas = (float)tau_best * (float)TIMESTEP;
    const float ratio  = T_meas / T_th;

    sim.Release();

    std::printf("  trace amp: rms=%.4f, peak=%.4f\n", (double)rms, (double)peak);
    std::printf("  measured T = %.3f s  vs  theoretical T = %.3f s  (ratio = %.3f)\n",
                (double)T_meas, (double)T_th, (double)ratio);

    EXPECT(rms > 0.005f, "wave amplitude survives (rms = %.4f > 0.005)", (double)rms);
    EXPECT(std::fabs(ratio - 1.f) < 0.10f,
           "measured period within 10%% of theoretical (got %.3f)", (double)ratio);
}

// ----------------------------------------------------------------------------
// Test 5: General stability sanity --- run a damaging dam-break scenario for
//         a few hundred steps, ensure no NaN/Inf appears anywhere.
// ----------------------------------------------------------------------------
static void TestDamBreakStability(int steps)
{
    std::printf("[Test 5] Dam-break stability over %d steps\n", steps);

    Sim sim;
    sim.ResetTerrain(0);
    sim.ResetWater(1, /*level=*/ 6.f);
    sim.SyncToHost();

    bool any_nan = false;
    for (int s = 0; s < steps; s++) {
        sim.SimStep(/*SWEonly=*/ false);
        if ((s % 60) == 59) {
            sim.SyncToHost();
            for (int x = 0; x < GRIDRESOLUTION; x++) {
                if (!std::isfinite(sim.h[x]) || !std::isfinite(sim.q[x])
                    || !std::isfinite(sim.htilde[x]) || !std::isfinite(sim.qtilde[x])) {
                    any_nan = true; break;
                }
            }
            if (any_nan) break;
        }
    }
    sim.Release();
    EXPECT(!any_nan, "simulation remained finite throughout");
}

int main(int argc, char** argv)
{
    // Best-effort: bind to device 0 if present.
    int devCount = 0;
    cudaError_t e = cudaGetDeviceCount(&devCount);
    if (e != cudaSuccess || devCount <= 0) {
        std::fprintf(stderr, "No CUDA device found (err = %d).\n", (int)e);
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(0));

    TestFftRoundTrip();
    TestStaticWaterConservation(/*steps=*/ 600);
    TestSWEOnlyBasicRun(/*steps=*/ 600);
    TestStandingWaveDispersion();
    TestDamBreakStability(/*steps=*/ 600);

    if (g_failures == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    } else {
        std::printf("\n%d test(s) failed.\n", g_failures);
        return 1;
    }
}
