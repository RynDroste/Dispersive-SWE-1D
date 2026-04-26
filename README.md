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

- Initialize `hbar = terrain + h`, `qbar = q`, and at cell edges $x+1/2$ compute
  $$
  \alpha_{\bar h} = \big(\sigma/\sigma_{\max}\big)^2 \cdot \exp(-d\,|\nabla h|^2)
  $$
- Explicit FTCS diffusion step solving
  $$
  \partial H/\partial T = \partial(\alpha\,\partial H/\partial x)/\partial x
  $$
- Recover $\bar h = \bar H - \tau$, $\tilde h = h - \bar h$, $\tilde q = q - \bar q$. On dry / steep-slope faces, both `qbar` and `qtilde` are zeroed simultaneously, equivalent to a reflective boundary.

### Airy Surface Wave Solver

Since SWE uses leapfrog with $\tilde h$ stored at $t+\Delta t/2$ and $\tilde q$ stored at $t$, eWave needs them at the same time instant. 

### Bulk SWE Solver

- Recover $\bar u = \bar q/\bar h$ from $\bar q$.
- For each face $x+1/2$, take three upwind face fluxes $\bar q^*$, average them to get cell-centered $\bar q^*_0,\,\bar q^*_{p1}$ and upwind-neighbor $\bar u^*_0,\,\bar u^*_{p1}$, then write

  $$
  \frac{d\bar u_{x+1/2}}{dt} = -\frac{1}{\bar h_{\text{face}}}\!\left(\frac{\bar q^*_{p1}\bar u^*_{p1}-\bar q^*_0\bar u^*_0}{\Delta x} - \bar u_{x+1/2}\,\frac{\bar q^*_{p1}-\bar q^*_0}{\Delta x}\right) - g\,\frac{\partial(\tau+\bar h)}{\partial x}
  $$

  where $\bar h_{\text{face}}=(\bar h_x+\bar h_{x+1})/2$. The result is then passed through `LimitVelocity_d`.
- Convert $\bar u^{t+\Delta t}$ back to $\bar q^{t+\Delta t} = \bar u^{t+\Delta t}\bar h^{t+\Delta t/2}$, **using the latest $\bar h$** here, again upwinded by sign.

### Transport Surface Flux and Surface Height with $\bar u$

- Use the half-step bulk velocity $\bar u^{t+\Delta t/2} = (\bar u^t+\bar u^{t+\Delta t})/2$ for semi-Lagrangian backtracking, calling `SampleCubicClamped_d` to fetch `qtilde_dummy` at $x-\Delta t\,\bar u^{t+\Delta t/2}$. Near dry cells.
- At each face, compute $\nabla\cdot\bar u$ via 2nd-order differencing of the average velocities at the two neighboring faces. Multiply the negative divergence by $\gamma=1/4$, then apply $\tilde q\,\mathrel{*}=\,\exp(-\nabla\cdot\bar u\cdot\Delta t)$. When the flow converges (shoaling amplification), it is dampened by a factor of 1/4; when it diverges, it decays at full rate.
- $\tilde h$ is at cell centers, using first-order differencing `ubarNew[x] - ubarNew[x-1]`. Same negative divergence × 0.25 + `exp(...)` treatment.

### Bulk-Advected Surface Displacement Update of $h$

- Use $\bar u^{t+\Delta t}$ to do a half-step semi-Lagrangian backtrack of $\tilde h$ to time $t+\Delta t$ at face positions, then multiply by $\bar u^{t+\Delta t}$ to obtain the flux $\breve q = \tilde{\bar h}\,\bar u$.
- For each cell center, pass each of the left and right faces through `Limit_flow_rate_d` once. When both sides are dry / the termination condition triggers, zero out the corresponding face flux, and finally `h ← max(0, h_dummy − dt·(q_r − q_l)/dx)`.

### Recombine Fluxes and Final Water Height Update

- $q = \mathrm{Limit}(\bar q + \tilde q,\, h_x,\, h_{x+1})$. Zeroed when `StopFlowOnTerrainBoundary_d` triggers or when `x == 0 || x ≥ N-2` (the first two / last two faces are reflective walls).
- $h\mathrel{+}= -\Delta t\,(q_x - q_{x-1})/\Delta x$, then clamp to 0.
- Pass through `Limit_flow_rate_d` once more with the updated $h$, providing a CFL-safe initial value for the next step.

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
