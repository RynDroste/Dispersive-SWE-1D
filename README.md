# Dispersive SWE 1D

Thanks to Dr. Stefan Jeschke, who has offer me a lot of help in contact during this project.

---

## 1. Core Idea Overview

1D implementation referencing the paper *"Generalizing Shallow Water Simulations with Dispersive Surface Waves"* (Jeschke & Wojtan, SIGGRAPH 2023).

This project is implemented on the GPU (CUDA).

- **Shallow Water Equations (SWE)** can simulate floods, inundation, and vortices as nonlinear bulk flow, but their dispersion relation $\omega = k\sqrt{gh}$ propagates all wavelengths at the same speed, so they cannot produce ship wakes or rise-and-fall interference patterns.
- **Airy linear wave theory** accurately reproduces the dispersion relation $\omega = \sqrt{gk\tanh(kh)}$, but only applies to small amplitudes and fixed water depth — it cannot handle domain deformation.

The solution is to perform a **bulk + surface decomposition** on the water height field and the flux field:

$$
h = \bar{h} + \tilde{h}, \quad q = \bar{q} + \tilde{q}
$$

The low-frequency components $\bar{h}, \bar{q}$ are solved with SWE (Stelling & Duinmeijer 2003), while the high-frequency components $\tilde{h}, \tilde{q}$ are solved in the frequency domain with the eWave (Tessendorf 2014) exponential integrator. The decomposition is redone every step, then the fluxes are recombined to update the water height, strictly conserving volume.

The overall structure is: decomposition → bulk SWE → Airy surface waves → transport surface quantities with $\bar{u}$ → recombine fluxes → update water height from flux divergence.

---

## 2. SimStep Main Loop

### Bulk vs. Surface Decomposition

- `kernel_init_decomp`: Initialize `hbar = terrain + h`, `qbar = q`, and at cell edges $x+1/2$ compute
  $$\alpha_{\bar h} = \big(\sigma/\sigma_{\max}\big)^2 \cdot \exp(-d\,|\nabla h|^2)$$
  where $\sigma = \min(\sigma_{\max}, \max(0,\,\bar H_{\text{face}} - \tau_{\max,\text{face}}))$ is the effective water depth at the face, $\sigma_{\max}=8$, $d=0.01$. The exponential term is a "gradient penalty" specifically designed to prevent steep waves like dam-break shocks from being incorrectly assigned to the Airy solution.
- `kernel_compute_alpha_qbar`: Average `alpha_hbar` over neighboring edges to obtain `alpha_qbar`, co-located with `qbar` (at cell centers).
- `kernel_diffusion_step`: Explicit FTCS diffusion step solving
  $$\partial H/\partial T = \partial(\alpha\,\partial H/\partial x)/\partial x$$
  The coefficient 0.48 < 0.5 (1D von Neumann upper bound). The last cell is only copied, not updated. Performs one 3-point update each on `hbar` and `qbar`, ensuring `hbar ≥ terrain`.
- `kernel_finalize_decomposition`: Recover $\bar h = \bar H - \tau$, $\tilde h = h - \bar h$, $\tilde q = q - \bar q$. On dry / steep-slope faces, both `qbar` and `qtilde` are zeroed simultaneously, equivalent to a reflective boundary.

### Airy Surface Wave Solver

Since SWE uses leapfrog with $\tilde h$ stored at $t+\Delta t/2$ and $\tilde q$ stored at $t$, eWave needs them at the same time instant. Full pipeline:

1. **`kernel_pack_fwd_input`**: Write $0.5\,(\tilde h^{t}+\tilde h^{t-\Delta t})$ into the first half of `fwd_buf_d` and $\tilde q^{t}$ into the second half. Simultaneously back up the current $\tilde h$ in place using `htildeOld_inout` for the next step. Each thread reads/writes only its own index, so there are no race conditions.
2. **`cufftExecC2C(plan_fwd, FORWARD)`**: A single forward transform with batch=2 obtains both $\hat{\tilde h}$ and $\hat{\tilde q}$ at once.
3. **`kernel_freq_domain`**: Each thread handles one $k$ and does four things:
   - Recover the signed $kS$ from the FFT bin and the physical $k=2\pi|kS|/(N\Delta x)$;
   - Multiply `htildehat` by $-iks$ (frequency-domain gradient, Johnson 2011), then by $e^{i\,kS\,\Delta x/2}$ for a half-cell phase shift, translating $\partial\hat{\tilde h}/\partial x$ to the face position where $\tilde q$ lives.
   - Loop over 4 sample depths:
     $$\omega_d = \sqrt{g\,k\,\tanh(k\,\text{Depth}_d)}$$
     $$1/\beta=1/\sqrt{2/(k_2\pi)\cdot\sin(k_2\pi/2)}$$
     fully cancels the numerical dispersion caused by finite-volume divergence;
   - Apply
     $$\hat{\tilde q}^{(d),\,t+\Delta t} = \cos(\omega_d\Delta t)\,\hat{\tilde q}^t - \sin(\omega_d\Delta t)\frac{\omega_d}{k^2}\Big(e^{i\,kS\,\Delta x/2}\,\partial_x\hat{\tilde h}\Big)$$
     and store in segment $d$ of `inv_buf_d`.
   `kNonZero = max(0.01, k)` and `k2 = max(1e-4, 2|kx|/N)` are NaN guards at $k\to 0$.
4. **`cufftExecC2C(plan_inv, INVERSE)`**: A single inverse transform with batch=`DEPTH_NUM` pulls all 4 segments back to physical space at once.
5. **`kernel_depth_interp_qtilde`**: For each cell, take `max(hbar[x], hbar[x+1])` as the local water depth and piecewise-linearly interpolate between the 4 depths.

### Bulk SWE Solver

- **`kernel_qbar_to_ubar`**: Recover $\bar u = \bar q/\bar h$ from $\bar q$, using the **previous step's** `hbarOld` and first-order upwinding based on the sign of $\bar q$ (in leapfrog, $\bar h$ and $\bar q$ are staggered by half a step in time, so the previous frame's $\bar h$ must be used). The result is then passed through `LimitVelocity_d`.
- **`kernel_swe_momentum`**: For each face $x+1/2$, take three upwind face fluxes $\bar q^*$ (i.e., `q_m05 / q_p05 / q_p15`), average them to get cell-centered $\bar q^*_0,\,\bar q^*_{p1}$ and upwind-neighbor $\bar u^*_0,\,\bar u^*_{p1}$, then write

  $$
  \frac{d\bar u_{x+1/2}}{dt} = -\frac{1}{\bar h_{\text{face}}}\!\left(\frac{\bar q^*_{p1}\bar u^*_{p1}-\bar q^*_0\bar u^*_0}{\Delta x} - \bar u_{x+1/2}\,\frac{\bar q^*_{p1}-\bar q^*_0}{\Delta x}\right) - g\,\frac{\partial(\tau+\bar h)}{\partial x}
  $$

  where $\bar h_{\text{face}}=(\bar h_x+\bar h_{x+1})/2$. The result is then passed through `LimitVelocity_d`. **This stage is the source of all nonlinearity in pure SWE** — floods, shocks, and shallow-water vortices all originate here.
- **`kernel_ubar_to_qbar`**: Convert $\bar u^{t+\Delta t}$ back to $\bar q^{t+\Delta t} = \bar u^{t+\Delta t}\bar h^{t+\Delta t/2}$, **using the latest $\bar h$** here, again upwinded by sign.

### Transport Surface Flux and Surface Height with $\bar u$

- **`kernel_advect_qtilde`** (preceded by `cudaMemcpyAsync(qtilde_dummy_d ← qtilde_d)`): Use the half-step bulk velocity $\bar u^{t+\Delta t/2} = (\bar u^t+\bar u^{t+\Delta t})/2$ for semi-Lagrangian backtracking, calling `SampleCubicClamped_d` to fetch `qtilde_dummy` at $x-\Delta t\,\bar u^{t+\Delta t/2}$. Near dry cells (where `h<0.01` on the side along the motion direction), zero out to prevent surface waves from being pushed into waterless regions. Corresponds to the $-\bar u\cdot\nabla\tilde q$ term in Eq. 15.
- **`kernel_div_ubar_qtilde`**: At each face, compute $\nabla\cdot\bar u$ via 2nd-order differencing of the average velocities at the two neighboring faces. Multiply the negative divergence by $\gamma=1/4$, then apply $\tilde q\,\mathrel{*}=\,\exp(-\nabla\cdot\bar u\cdot\Delta t)$. When the flow converges (shoaling amplification), it is dampened by a factor of 1/4; when it diverges, it decays at full rate.
- **`kernel_div_ubar_htilde`**: $\tilde h$ is at cell centers, using first-order differencing `ubarNew[x] - ubarNew[x-1]`. Same negative divergence × 0.25 + `exp(...)` treatment.

### Bulk-Advected Surface Displacement Update of $h$

- **`kernel_compute_advectHFR`**: Use $\bar u^{t+\Delta t}$ to do a half-step semi-Lagrangian backtrack of $\tilde h$ to time $t+\Delta t$ at face positions, then multiply by $\bar u^{t+\Delta t}$ to obtain the flux $\breve q = \tilde{\bar h}\,\bar u$ (sample position `x + 0.5 - 0.5·dt·u`: the `+0.5` is because the target is at face $x+1/2$, and `-0.5·dt·u` pulls $\tilde h$ back half a step from $t+3\Delta t/2$ to $t+\Delta t$).
- **`kernel_h_update_from_advectHFR`** (preceded by `cudaMemcpyAsync(h_dummy_d ← h_d)`): For each cell center, pass each of the left and right faces through `Limit_flow_rate_d` once. When both sides are dry / the termination condition triggers, zero out the corresponding face flux, and finally `h ← max(0, h_dummy − dt·(q_r − q_l)/dx)`. This step deducts the transport flux from $h$ first; the remaining bulk + surface portions are deducted again. The linearity of divergence guarantees that the sum of the two steps equals one complete divergence update.

### Recombine Fluxes and Final Water Height Update

- **`kernel_recombine_q`**: $q = \mathrm{Limit}(\bar q + \tilde q,\, h_x,\, h_{x+1})$. Zeroed when `StopFlowOnTerrainBoundary_d` triggers or when `x == 0 || x ≥ N-2` (the first two / last two faces are reflective walls).
- **`kernel_height_integration`**: $h\mathrel{+}= -\Delta t\,(q_x - q_{x-1})/\Delta x$, then clamp to 0. **Water height is always derived from flux divergence → strict volume conservation**.
- **`kernel_q_final_limit`**: Pass through `Limit_flow_rate_d` once more with the updated $h$, providing a CFL-safe initial value for the next step.

---

## 3. Scenes

`ResetTerrain` provides two terrain types: flat ground and hills.

---

## 4. Dependencies

- **CUDA Toolkit (≥ 11.0)**
- ImGui

---

## 5. References

- Jeschke, S. and Wojtan, C. *Generalizing Shallow Water Simulations with Dispersive Surface Waves.* ACM SIGGRAPH 2023.
- Stelling, G. S. and Duinmeijer, S. P. A. *A staggered conservative scheme for every Froude number in rapidly varied shallow water flows.* Int. J. Numer. Methods Fluids, 2003.
- Tessendorf, J. *eWave: Using an Exponential Solver on the iWave Problem.* Tech. Note, 2014.
- Johnson, S. G. *Notes on FFT-based differentiation.* MIT Tech. Rep., 2011.
