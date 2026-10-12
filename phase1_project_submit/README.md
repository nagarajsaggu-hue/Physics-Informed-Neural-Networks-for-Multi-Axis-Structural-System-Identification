<p align="center">
  <img src="assets/banner.png" alt="PINN multi-axis structural system identification — project overview" width="100%">
</p>

<h1 align="center">Physics-Informed Neural Networks for Multi-Axis Structural System Identification</h1>

<p align="center">
  <em>A direction-agnostic framework for stiffness and damping recovery from single-node displacement records</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/phase-1%20complete-B5174A?style=flat-square" alt="Phase 1 complete">
  <img src="https://img.shields.io/badge/python-3.12-3F6E5A?style=flat-square" alt="Python 3.12">
  <img src="https://img.shields.io/badge/DeepXDE-PyTorch%20backend-4A5D73?style=flat-square" alt="DeepXDE + PyTorch">
  <img src="https://img.shields.io/badge/data-ANSYS%20%2F%20PyMAPDL-C68A2E?style=flat-square" alt="ANSYS / PyMAPDL">
</p>

<p align="center">
  <b>Nagaraju Saggu</b> · Supervisor: Rohan Raj Das<br>
  Chair of Optimization and Stochastic Structural Engineering · Bauhaus-Universität Weimar · September 2026
</p>

---

## Contents

1. [In one paragraph](#in-one-paragraph)
2. [The problem](#the-problem)
3. [The core idea](#the-core-idea)
4. [The data](#the-data)
5. [Method](#method)
6. [Results](#results)
7. [The key finding: when is damping observable?](#the-key-finding-when-is-damping-observable)
8. [Repository structure](#repository-structure)
9. [Running the pipeline](#running-the-pipeline)
10. [Limitations](#limitations)
11. [Future work (Phase 2)](#future-work-phase-2)
12. [References](#references)

---

## In one paragraph

A steel support table (78,500 kg) is shaken by a horizontal 15 Hz harmonic force in an ANSYS transient analysis. The displacement of **one node (Node 3575)** is recorded in X, Y and Z. A **physics-informed neural network (PINN)** learns a smooth surrogate of that displacement while simultaneously identifying the **stiffness K** and **damping C** of an equivalent single-degree-of-freedom oscillator for the chosen axis. Only displacement is used for training: velocity and acceleration come from exact automatic differentiation of the network, never from differentiating the data. A single configuration-driven code path runs unchanged on all three axes. Stiffness is recovered in every direction to within **0.4 %** of an independent least-squares estimator; damping is recovered where the record actually contains the information needed to determine it, and the method's own convergence diagnostics show where it does not.

---

## The problem

The forward problem of structural dynamics is settled: given a structure, compute its response. Structural health monitoring needs the **inverse**: given a measured response, what are the stiffness and damping of the structure that produced it?

For a single-degree-of-freedom idealisation,

$$
M\ddot u(t) + C\dot u(t) + K u(t) = F(t)
$$

the classical approach rearranges this into a linear least-squares problem in $K$ and $C$. Its weakness is structural: it needs $\dot u$ and $\ddot u$, which are not measured and must be obtained by **numerically differentiating** the displacement record. Differentiation amplifies high-frequency content (gain grows as $\omega^2$), is sensitive to sampling rate and endpoints, and the estimator returns a number whether or not the data actually determine the parameter.

## The core idea

Replace the sampled record with a **continuous neural surrogate** $\hat u_\theta(t)$, differentiate *that* exactly with automatic differentiation, and require it to satisfy the equation of motion at thousands of collocation points. Promote $K$ and $C$ to trainable scalars.

| Structural dynamics | Physics-informed learning |
|---|---|
| Equation of motion $M\ddot u + C\dot u + Ku = F$ | Physics residual penalty $\mathcal L_{\text{phys}}$ |
| Measured displacement at one node | Data misfit $\mathcal L_{\text{data}}$ |
| Velocity and acceleration (not measured) | Automatic differentiation of the surrogate |
| Passive structure dissipates: $C > 0$ | Softplus reparameterisation |
| Single-mode reduction, generalised mass $M_u$ | Known constant in the residual |
| Oscillatory response, ~190 cycles per record | Fourier features overcome spectral bias |
| Damping observable only if $T \gtrsim \tau = 2M/C$ | Vanishing gradient $\partial\mathcal L/\partial C \to 0$ |

The crucial structural fact: **$K$ and $C$ appear only in the physics term.**

$$
\frac{\partial \mathcal L_{\text{data}}}{\partial K} = \frac{\partial \mathcal L_{\text{data}}}{\partial C} = 0
$$

A network that merely memorises the displacement receives no gradient on $K$ or $C$. Fitting the curve is necessary but not sufficient — the identified parameters must be checked independently, which is why every result below is compared against two separate references.

---

## The data

| | X | Y | Z |
|---|---|---|---|
| Source | ANSYS transient analysis | ANSYS transient analysis | ANSYS transient analysis |
| Observation | Node 3575, `Displacement_X (m)` | Node 3575, `Displacement_Y (m)` | Node 3575, `Displacement_Z (m)` |
| Samples / span | 1,000 / 0.01–10 s | 1,000 / 0.01–10 s | 10,000 / 0.001–10 s |
| Sampling rate | 100 Hz | 100 Hz | 1 kHz |
| Drive | 15 Hz harmonic | 15 Hz harmonic | 15 Hz harmonic |
| Reference mode (modal analysis) | 1 (20.08 Hz) | 2 (20.66 Hz) | 4 (51.43 Hz) |
| Generalised mass $M_u$ | 56,767 kg | 57,486 kg | 5,736,762 kg |

Each CSV also carries velocity, acceleration and force columns. **Only displacement is used for training.** The exported kinematic columns are not mutually consistent derivatives of the displacement (correlation 0.88 in X/Y), and fitting against them produced unphysical negative damping. They are kept only as diagnostic cross-checks.

<details>
<summary><b>Why is the Z generalised mass 73× the mass of the whole table?</b></summary>

<br>

The single-mode reduction expresses the oscillator in the coordinate of the observed node:

$$
M_u = \frac{M_r}{\phi_{r,j}^2}, \qquad K_u = \frac{K_r}{\phi_{r,j}^2}
$$

With mass-normalised modes ($M_r = 1$), $M_u = 1/\phi_{r,j}^2$. Node 3575 sits near a nodal region of mode 4 in Z, so $\phi_{4,j} = 4.175\times10^{-4}$ — ten times smaller than in Y — and the generalised mass is a hundred times larger. It is a coordinate-dependent scaling factor, not a physical mass. The invariant that can be compared across axes is $K_u/M_u = \omega_r^2$. Both identities were verified to full recorded precision against the ANSYS sidecar files.

</details>

---

## Method

### Architecture

```
 t ──► Fourier features ──► 4 × 64 sine MLP ──► û_θ(t) ──┬──► data loss  ‖û − u_meas‖²
      128 linear freqs       Glorot uniform              │
      5–40 Hz (Z: 5–60 Hz)                               └──► autodiff ──► u̇, ü ──► physics residual
                                                                              M ü + C u̇ + K u − F(t)
                                       K = softplus(ρ_k),  C = softplus(ρ_c)  ──────────┘  (trainable)
```

### Composite loss

$$
\mathcal L(\theta, K, C) = w_d \,\frac{1}{N_d}\sum_i \big(\hat u_\theta(t_i) - u_i\big)^2 \;+\; w_f\,\frac{1}{N_f}\sum_j \big(M\ddot{\hat u}_\theta + C\dot{\hat u}_\theta + K\hat u_\theta - F\big)^2(t_j)
$$

with $w_d = 100$, $w_f = 1$.

### Design decisions, and why

| Decision | Reason |
|---|---|
| **Fourier feature encoding** (128 frequencies, band bracketing the response spectrum) | A 10 s record holds ~190 oscillations. Spectral bias makes a raw-coordinate network fail beyond ~30 cycles (it collapses to zero output). The encoding moves the learnable band onto the response frequencies. Z uses 5–60 Hz because its reference mode is at 51.4 Hz. |
| **Sine activation** (SIREN-style) | Derivatives of sines are sines: differentiation does not degrade the function class, and the target is oscillatory. |
| **Non-dimensionalisation** $\hat u = u/u_{\text{ref}}$, $\hat F = F/F_{\text{ref}}$ | Raw magnitudes span 10⁻⁷ m to 10¹¹ N/m. Scaling puts $\hat k$ in the range 1–8 so one learning rate serves weights and physical parameters. Time is not rescaled. |
| **Softplus reparameterisation** $C = \ln(1+e^{\rho})$ | Makes negative damping unreachable rather than penalised. Initialised through the inverse softplus so configured starting values mean what they say. |
| **Explicit holdout** (sample $i$ held out iff $i \bmod 5 = 2$) | DeepXDE's "test" loss is drawn from the training distribution. 20 % of samples are withheld from the loss entirely. |
| **Parameter logging every 100 iterations** | Under Adam a weakly constrained parameter *drifts* instead of stalling, so the trajectory carries information the final value does not. |

### Training

Adam, learning rate 10⁻³, 30,000 iterations, no schedule, no L-BFGS stage. 10,000 interior collocation points plus every training time as an anchor. Seed 0 for production. Each axis ran as one batch job on an NVIDIA A100 in under 4.5 minutes of training.

### Independent references

1. **Classical least squares** on spectrally differentiated displacement (5 % endpoint trim), evaluated under **both** the recorded force column and the same analytic harmonic the PINN uses. Only the matched-force comparison is used to claim agreement.
2. **ANSYS modal analysis** — natural frequency and generalised stiffness of the selected reference mode.

---

## Results

### Surrogate accuracy (held-out samples never seen in training)

| Metric | X | Y | Z |
|---|---|---|---|
| R², held out | 0.99992 | 0.99989 | 0.99999 |
| Relative RMSE, held out | 0.88 % | 1.06 % | 0.32 % |
| Held-out / training RMSE | 1.14 | 1.51 | 0.98 |

No sign of memorisation. These numbers measure interpolation quality, not parameter correctness.

### Identified parameters

| | X | Y | Z |
|---|---|---|---|
| Identified K (N/m) | 8.183 × 10⁸ | 8.803 × 10⁸ | 5.992 × 10¹¹ |
| Identified C (N·s/m) | 3.409 × 10⁴ | 3.438 × 10⁴ | 2.35 × 10³ *(not converged)* |
| Natural frequency $f_n$ | 19.11 Hz | 19.69 Hz | 51.44 Hz |
| Damping ratio ζ | 0.250 % | 0.242 % | — |
| Decay time constant τ = 2M/C | 3.33 s | 3.34 s | ~4,875 s |

### Agreement with the independent references

| Comparison | X | Y | Z |
|---|---|---|---|
| **K vs matched-force least squares** | **−0.28 %** | **−0.38 %** | **0.015 %** |
| C vs matched-force least squares | +20.4 % (LS std. error 38 %) | ≈ +6 % (LS std. error 38 %) | LS returns C < 0 |
| K vs ANSYS modal stiffness | −9.5 % | −9.1 % | +0.014 % |

**Stiffness** is the principal result: two methods sharing no machinery — a neural surrogate trained by gradient descent on displacement alone, and closed-form regression on spectral derivatives — agree to within 0.4 % on three axes spanning three orders of magnitude in stiffness.

The consistent ~9 % gap to the modal reference in X and Y is most likely a model-configuration mismatch: the transient solve predates the modal analysis, and the identified frequencies sit closer to the secondary response peaks (17.9 and 18.3 Hz) than to the modal lines. The Z modal agreement is a consistency check rather than an independent validation, because the Z generalised force amplitude (95,073 N) was configured rather than derived from the load vector — establishing it from first principles is the first Phase 2 task.

### Convergence

| | X | Y | Z |
|---|---|---|---|
| K travel from initialisation | × 7.37 | × 7.93 | × 1.26 |
| K range, last 5,000 iterations | 0.087 % | 0.069 % | 0.012 % |
| C range, last 5,000 iterations | 0.52 % | 0.38 % | **24.4 %, still drifting** |

Stiffness settles tightly everywhere. Damping settles in X and Y and keeps drifting in Z — which is exactly what the theory predicts.

---

## The key finding: when is damping observable?

The free response decays as $e^{-t/\tau}$ with $\tau = 2M/C$. Over a record of length $T$ the amplitude falls by $1 - e^{-T/\tau}$. Damping therefore enters the data, to leading order, only through the ratio $T/\tau$.

| | X | Y | Z |
|---|---|---|---|
| $T/\tau$ | 3.00 | 2.99 | 0.002 |
| Envelope decay over the record | 95 % | 95 % | **0.20 %** |
| PINN damping | converged | converged | drifting |
| Unconstrained least-squares damping | positive | positive | **negative** |

In Z the envelope decays by 0.20 % across the entire record — less than the surrogate's own 0.33 % fit error. The information needed to determine $C_Z$ is simply not in the data, and no estimator can extract it. The PINN shows this as continued drift; the unconstrained classical estimator shows it as a negative, unphysical value. Both are the same condition expressed two ways.

Stiffness behaves completely differently. A change $\delta K$ shifts the oscillation frequency and accumulates phase error $\delta\varphi = \tfrac{\omega_n t}{2}\tfrac{\delta K}{K}$, visible within a few cycles. A window-length study on X confirmed this: a 0.5 s window (9 cycles) already gives K to within 0.22 %, while damping from that window is meaningless in both estimators.

> **Stiffness is identifiable from a few cycles. Damping is identifiable only when the record is long compared with its decay time.** A method whose diagnostics distinguish these two cases is more useful than one that returns a confident number for every parameter.

---

## Repository structure

```
.
├── config.yaml              # one active direction block (X, Y or Z) at a time
├── AGENTS.md                # working notes and conventions
├── data/                    # ANSYS exports (do not edit — regenerated in ANSYS)
├── src/sysid/
│   ├── config.py            # YAML/JSON loading, active-direction resolution
│   ├── dataio.py            # CSV ingestion, windowing, non-dimensionalisation
│   ├── physics.py           # equation of motion, scaling, softplus, residual, least squares
│   ├── train.py             # network, Fourier encoding, holdout, training loop
│   ├── evaluate.py          # comparison against least-squares and modal references
│   └── viz.py               # figures
├── scripts/
│   ├── run_train.py
│   ├── run_evaluate.py
│   └── run_plot.py
├── tests/
│   ├── test_dataio.py
│   └── test_physics.py
├── outputs/phase1/{X,Y,Z}/  # per-direction results, convergence logs, figures/
└── assets/banner.png
```

## Running the pipeline

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. choose a direction: in config.yaml, uncomment exactly one block (X, Y or Z)

# 3. tests, then train → evaluate → plot
pytest tests/
python scripts/run_train.py    --tag phase1
python scripts/run_evaluate.py --tag phase1
python scripts/run_plot.py     --tag phase1
```

Outputs land in `outputs/phase1/<direction>/`: identified parameters and all metrics in one JSON, per-sample predictions, the convergence history (301 logged states), and figures. Switching axes needs no code change — only the active block in `config.yaml`.

---

## Limitations

- **Single-mode reduction.** Secondary spectral content reaches 26 % (X), 24 % (Y) and 11 % (Z) of the drive amplitude. Identified values describe the best single-mode fit of the record, not the full structure.
- **Holdout is interpolation.** Held-out points are interleaved within the same record — not a forecast, another node, or another load case.
- **Surrogate accuracy ≠ physical adequacy.** The exact single-DOF solution with the identified parameters reproduces the record at 13–32 % relative RMSE, far worse than the network. The evidence for the parameters comes from the independent comparisons, not the fit.
- **Zero structural damping in the ANSYS source.** The identified C describes dissipation present in this dataset, not a material property of the steel.
- **Noise-free data.** All records are exact finite element output.

## Future work (Phase 2)

1. **Derive the generalised force from first principles**, $F_u(t) = \boldsymbol\phi_r^{\mathsf T}\mathbf f(t)/\phi_{r,j}$, turning the Z modal agreement into an independent validation.
2. **Re-solve the transient and modal analyses in one consistent configuration** to resolve the ~9 % X/Y modal gap.
3. **Free-decay (ring-down) analysis for Z damping**, since a forced record would need $T \sim \tau \approx 4{,}875$ s.
4. **Two- and three-mode formulations** with a vector residual, which would also make Rayleigh α and β separately identifiable.
5. **Noise-robustness sweep** to find the signal-to-noise ratio at which identification degrades.
6. **Multi-seed ensembles** on all three axes for empirical uncertainty bounds (a single seed-7 repeat in X reproduced K to 0.27 %, while C varied by ~30 %).

---

## References

- Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Lu, L., Meng, X., Mao, Z., Karniadakis, G. E. (2021). DeepXDE: a deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.
- Tancik, M. et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS 33*.
- Rahaman, N. et al. (2019). On the spectral bias of neural networks. *ICML*.
- Wang, S., Wang, H., Perdikaris, P. (2021). On the eigenvector bias of Fourier feature networks. *CMAME*, 384, 113938.
- Sitzmann, V. et al. (2020). Implicit neural representations with periodic activation functions. *NeurIPS 33*.
- Jacot, A., Gabriel, F., Hongler, C. (2018). Neural tangent kernel. *NeurIPS 31*.
- Kingma, D. P., Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.
- Chopra, A. K. (2017). *Dynamics of Structures*, 5th ed. Pearson.

The full Phase 1 report, with derivations and every figure, is in [`Pinn-Phase-1_Report_git.pdf`](Pinn-Phase-1_Report_git.pdf).

---

<p align="center"><sub>Bauhaus-Universität Weimar · Chair of Optimization and Stochastic Structural Engineering</sub></p>
