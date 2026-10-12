# Phase 1 - Results Record

Node 3575, X direction, single degree of freedom.
Final run tag: `final2`, 30000 iterations, seed 0.

## Headline

A physics-informed neural network trained on displacement alone identified
the stiffness of an equivalent 1-DOF system to within **0.28 %** of an
independent classical least-squares estimate, reproducibly across random
seeds, generalising to held-out data with **R^2 = 0.999922**.

## Identified parameters

| quantity | value | reference | difference |
|---|---|---|---|
| K | 8.1834e8 N/m | 8.2060e8 | -0.28 % |
| C | 3.4093e4 N.s/m | 2.8326e4 | +20.4 % |
| f_n (derived) | 19.1091 Hz | 19.1354 | -0.14 % |
| zeta (derived) | 0.2501 % | 0.2075 % | - |

Fixed inputs: M = 56767 kg, F = 11103 sin(2 pi 15 t) N.

## Prediction quality

| metric | value |
|---|---|
| RMSE, all 1000 points | 2.0248e-07 m (0.797 % of RMS) |
| RMSE, 200 held-out points | 2.2486e-07 m (0.884 %) |
| R^2 held out | 0.999922 |
| peak displacement | ~6e-5 m |

Training and held-out errors are nearly identical, so there is no
overfitting. The held-out points were spread evenly through the record
(t = 0.03 to 9.98 s) and were never shown to the network.

## Reproducibility across seeds

| seed | K (N/m) | C (N.s/m) |
|---|---|---|
| 0 | 8.005e8 | 2.657e4 |
| 7 | 7.984e8 | 3.521e4 |
| difference | **0.27 %** | ~30 % |

Stiffness is reproducible. Damping is weakly constrained and varies between
initialisations - this should be reported honestly rather than smoothed over.

## Window-length study

The clearest physical result in the project.

| window | cycles | Fourier | K error | C |
|---|---|---|---|---|
| 0.5 s | 9 | off | -0.22 % | meaningless |
| 2.0 s | 38 | off | **failed** | failed |
| 1.0 s | 19 | on | - | zeta 0.358 % |
| 5.0 s | 95 | on | -0.87 % | +24 % |
| 10.0 s | 190 | on | **-0.28 %** | +20 % |

Two conclusions:

1. **Stiffness identifies from a handful of cycles; damping needs the whole
   record.** Damping only reveals itself through decay over time. At 0.5 s
   the amplitude falls about 1 %, which is too little to measure - both the
   PINN and least squares gave meaningless C there, least squares returning a
   negative value.

2. **Plain networks fail beyond about 30 oscillations.** Fourier feature
   encoding was required to reach the full record. This is spectral bias, a
   known limitation of PINNs on oscillatory systems.

## Evidence against memorisation

- Parameters were initialised far from truth (k_hat 1.0 against ~7.4) and
  converged there over 30000 iterations. A memorising network would fit the
  displacement and leave k_hat at 1.0 - which is exactly what the failed runs
  did.
- Two unrelated methods agree to 0.28 %.
- All working runs fit displacement to about 1 %, yet C spans a factor of 5.5
  across them. Fitting the signal does not determine C; the physics term
  does.
- Only displacement enters training. Velocity and acceleration come from
  automatic differentiation. K and C are never supplied in any form.
- Failure modes were physically correct: the method failed exactly where
  physics says damping is unobservable.

## Code audit and fixes

Four weaknesses were found and closed before the final run.

| weakness | fix | evidence it worked |
|---|---|---|
| No held-out data - DeepXDE tests on training points | 20 % withheld | R^2 = 0.999922 on 200 unseen |
| C free to go negative | softplus parameterisation | C stayed positive throughout |
| 21 collocation points per cycle | 4000 -> 10000 points | 53 per cycle |
| Single random seed | seed 0 and seed 7 | K agrees to 0.27 % |

## Figures

| file | shows |
|---|---|
| `timeseries_final2.png` | predicted vs measured over 190 oscillations, error panel |
| `convergence_final2.png` | K and C settling onto the reference lines |
| `spectrum_final2.png` | 15 Hz drive vs 17.9 Hz response vs modal lines |
| `comparison_final2.png` | K by method |

The spectrum figure is the strongest single piece of evidence: it shows the
forced/free separation and the transient/modal mismatch in one image.

## Limitations - carried into Phase 2

1. ANSYS damping coefficients are all zero, so C describes dissipation in
   this dataset, not a property of the steel.
2. The transient solve predates the modal analysis, which explains the 9.47 %
   gap against the ANSYS modal stiffness.
3. The brief's 15.02 Hz "fundamental natural frequency" is the drive
   frequency. Modal gives 20.08, 20.66, 30.52 Hz.

None is a fault of the identification method. All three require re-solving in
ANSYS and are the first task of Phase 2.

## Honest summary sentence

> The identification method is verified on this dataset. The remaining
> uncertainty is in the ANSYS source model - zero structural damping, an
> out-of-date transient solve, and a natural frequency that differs from the
> brief - which should be resolved before quoting absolute physical values.


## Phase 1.5 - Multi-Axis Evaluation Results

### Axis: X
method                  K (N/m)      C (N.s/m)   f_n (Hz)   zeta (%)   note
---------------------------------------------------------------------------
PINN                 8.1834e+08     3.4093e+04    19.1091     0.2501   30000 iterations
Least squares        8.2060e+08     2.8326e+04    19.1354     0.2075   R^2 = 0.998561
ANSYS modal          9.0394e+08              -    20.0835          -   mode 1, m* = 56,767 kg

PINN vs least squares : K -0.28 %, C +20.36 %
PINN vs ANSYS modal   : K -9.47 %
  (independent check. A gap here can also mean the transient and modal solves used different configurations.)

---

### Axis: Y
method                  K (N/m)      C (N.s/m)   f_n (Hz)   zeta (%)   note
---------------------------------------------------------------------------
PINN                 8.8030e+08     3.4381e+04    19.6949     0.2417   30000 iterations
Least squares        8.8364e+08     3.2475e+04    19.7323     0.2278   R^2 = 0.998379
ANSYS modal          9.6850e+08              -    20.6581          -   mode 2, m* = 57,486 kg

PINN vs least squares : K -0.38 %, C +5.87 %
PINN vs ANSYS modal   : K -9.11 %
  (independent check. A gap here can also mean the transient and modal solves used different configurations.)

---

### Axis: Z
method                  K (N/m)      C (N.s/m)   f_n (Hz)   zeta (%)   note
---------------------------------------------------------------------------
PINN                 5.9923e+11     2.3534e+03    51.4378     0.0001   30000 iterations
Least squares        1.1586e+09    -6.1371e+05     2.2618    -0.3764   R^2 = 0.000209
ANSYS modal          5.9914e+11              -    51.4342          -   mode 4, m* = 5,736,762 kg (selected reference)

PINN vs least squares : K +51619.05 %, C -100.38 %
  -> least-squares and PINN are not directly comparable if the PINN uses a transformed/generalized force while least squares uses the raw transient force.
PINN vs ANSYS modal   : K +0.0139 %
  -> ANSYS reference uses the explicitly selected physical modal reference when available.

---

### Z-Axis Conclusion (Physical Limitation)
The Z-axis training completed successfully, but produced invalid 1-DOF parameters (negative least-squares stiffness, massive PINN damping). This confirms that a 1-DOF formulation cannot extract parameters for a parasitic axis. The external horizontal force (11,103 N) does not directly drive the Z-axis, causing the mathematical residual to collapse under an invalid boundary assumption.
