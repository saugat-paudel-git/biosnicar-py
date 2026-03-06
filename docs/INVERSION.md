# Inversion

The inverse module retrieves ice physical properties from observed albedo using an emulator-powered optimisation framework. Given spectral or satellite-band observations, it estimates bubble radius, density, and impurity concentrations with uncertainty.

## Quick Start

### Spectral retrieval (480-band)

```python
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")

# observed = your 480-band spectral albedo measurement
result = retrieve(
    observed=observed,
    parameters=["rds", "rho", "black_carbon", "glacier_algae", "dust"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50},
)
print(result.summary())
```

### Satellite band retrieval

```python
import numpy as np

# Band mode: fix density and sky conditions, retrieve grain size + impurities
result = retrieve(
    observed=np.array([0.82, 0.78, 0.75, 0.45, 0.03]),
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    obs_uncertainty=np.array([0.02, 0.02, 0.02, 0.03, 0.05]),
    fixed_params={"direct": 1, "solzen": 50, "rho": 600, "dust": 1000},
)
print(result.summary())
```

## API Reference

### `retrieve()`

Main entry point for parameter retrieval.

| Parameter             | Type          | Default      | Description                                                                            |
| --------------------- | ------------- | ------------ | -------------------------------------------------------------------------------------- |
| `observed`            | ndarray       | required     | 480-band spectral albedo or N-band satellite array                                     |
| `parameters`          | list[str]     | required     | Names of parameters to retrieve                                                        |
| `emulator`            | Emulator      | None         | Trained emulator (fast mode)                                                           |
| `forward_fn`          | callable      | None         | Direct forward function (slow mode). Either `emulator` or `forward_fn` required.       |
| `platform`            | str           | None         | Satellite platform key (e.g. `"sentinel2"`). Activates band mode.                      |
| `observed_band_names` | list[str]     | None         | Band names matching `observed` (e.g. `["B3", "B4", "B11"]`). Required with `platform`. |
| `obs_uncertainty`     | ndarray       | None         | Per-element 1-sigma measurement uncertainty                                            |
| `bounds`              | dict          | None         | `{name: (lo, hi)}` overriding `DEFAULT_BOUNDS`                                         |
| `x0`                  | dict          | None         | `{name: value}` overriding `DEFAULT_X0`                                                |
| `regularization`      | dict          | None         | `{name: (prior_mean, prior_sigma)}` Gaussian priors                                    |
| `wavelength_mask`     | ndarray[bool] | None         | Wavelength mask for spectral mode (True = include)                                     |
| `method`              | str           | `"L-BFGS-B"` | `"L-BFGS-B"`, `"Nelder-Mead"`, `"differential_evolution"`, or `"mcmc"`                 |
| `mcmc_walkers`        | int           | 32           | MCMC walkers (only for `method="mcmc"`)                                                |
| `mcmc_steps`          | int           | 2000         | MCMC steps                                                                             |
| `mcmc_burn`           | int           | 500          | MCMC burn-in steps to discard                                                          |
| `fixed_params`        | dict          | None         | `{name: value}` for known parameters not to be optimised                               |

**Returns:** `RetrievalResult`

### `RetrievalResult`

| Attribute             | Type            | Description                                                             |
| --------------------- | --------------- | ----------------------------------------------------------------------- |
| `best_fit`            | dict            | `{param_name: optimal_value}`                                           |
| `cost`                | float           | Final chi-squared value                                                 |
| `uncertainty`         | dict            | `{param_name: 1_sigma}` from Hessian or MCMC                           |
| `predicted_albedo`    | ndarray (480,)  | Spectrum at best-fit point                                              |
| `observed`            | ndarray         | Input observations                                                      |
| `converged`           | bool            | Optimiser convergence flag                                              |
| `method`              | str             | Optimisation method used                                                |
| `n_function_evals`    | int             | Number of forward evaluations                                           |
| `chains`              | ndarray or None | MCMC chains `(n_steps, n_walkers, n_params)`. Only for `method="mcmc"`. |
| `acceptance_fraction` | float or None   | MCMC acceptance rate                                                    |
| `autocorr_time`       | ndarray or None | MCMC autocorrelation time per parameter                                 |

`result.summary()` returns a human-readable string.

### Default Bounds and Initial Guesses

| Parameter       | Bounds        | Default x0 | Units                        |
| --------------- | ------------- | ---------- | ---------------------------- |
| `rds`           | (100, 5000)   | 1000       | um (bubble effective radius) |
| `rho`           | (100, 917)    | 500        | kg/m³ (density)              |
| `solzen`        | (20, 80)      | 50         | degrees (solar zenith angle) |
| `direct`        | (0, 1)        | 1          | binary (0=cloudy, 1=clear)   |
| `black_carbon`  | (0, 100000)   | 100        | ppb (black carbon)           |
| `snow_algae`    | (0, 500000)   | 10000      | cells/mL (snow algae)        |
| `glacier_algae` | (0, 100000)   | 100        | cells/mL (glacier algae)     |
| `dust`          | (0, 500000)   | 100        | ppb (mineral dust)           |

When using an emulator, bounds are auto-filled from the emulator's training range for parameters it knows.

**Note:** `direct` is a binary flag and **cannot** be included in `parameters` — it raises a `ValueError`. Pass it via `fixed_params` instead (see [Binary Parameters](#binary-parameters)).

### Cost Functions

- **`spectral_cost`**: Chi-squared over 480-band spectral albedo, with optional per-wavelength uncertainty, wavelength mask, and regularization
- **`band_cost`**: Chi-squared over satellite bands. Predicts 480-band albedo, convolves via `to_platform()`, compares to observed band values.

## Observation Modes

### Spectral mode (default)

Pass a 480-element array of spectral albedo (e.g. from a field spectrometer). All wavelengths are compared against the emulator prediction. Use `wavelength_mask` to exclude noisy bands (e.g. water vapour absorption at 1.38 and 1.87 um).

This mode provides the maximum information content and can typically constrain 4-5 parameters simultaneously.

### Satellite band mode

Set `platform` and `observed_band_names` to work natively in band space. The emulator predicts the full 480-band spectrum internally, convolves to the platform's bands, and compares to your observations. No need to reconstruct a continuous spectrum from satellite data.

Supported platforms: `sentinel2`, `sentinel3`, `landsat8`, `modis`.

**Important:** Band mode provides substantially less spectral information than spectral mode (typically 4-5 broadband values vs 480 spectral values). It cannot constrain as many free parameters. For robust band-mode retrieval:
- Fix well-constrained parameters (`direct`, `solzen`, `rho`, `dust`) via `fixed_params`
- Limit free parameters to 2-3 (typically grain radius plus 1-2 dominant impurities)
- Use `obs_uncertainty` to weight bands appropriately

### Direct model mode

Pass a `forward_fn` callable instead of an emulator. Slower (~seconds per retrieval) but uses the exact forward model. Useful for validating emulator-based results.

```python
from biosnicar import run_model
import numpy as np

result = retrieve(
    observed=observed,
    parameters=["rds"],
    forward_fn=lambda rds: np.array(run_model(rds=int(rds), layer_type=1).albedo),
    bounds={"rds": (100, 5000)},
)
```

## Retrievable Parameters

| Parameter       | Physical meaning            | Spectral effect                                                   | Sensitivity      |
| --------------- | --------------------------- | ----------------------------------------------------------------- | ---------------- |
| `rds`           | Bubble effective radius     | Controls NIR albedo: larger bubbles = less scattering = lower NIR | High (NIR)       |
| `rho`           | Ice density                 | Controls optical depth per unit thickness                         | Moderate         |
| `solzen`        | Solar zenith angle          | Path length effect; higher SZA = lower albedo                     | Moderate         |
| `black_carbon`  | Black carbon (upper layer)  | Broadband darkening, strongest in visible                         | High (VIS)       |
| `snow_algae`    | Snow algae (upper layer)    | Visible darkening with carotenoid and chl-a features              | High (VIS)       |
| `glacier_algae` | Glacier algae (upper layer) | Visible darkening with carotenoid and chl-a features              | High (VIS)       |
| `dust`          | Mineral dust (upper layer)  | Weak broadband visible darkening                                  | Low (see caveat) |

The impurity names correspond to the YAML keys in `inputs.yaml`. The defaults are `black_carbon`, `snow_algae`, `glacier_algae`, and `dust`.

### Dust sensitivity caveat

Mineral dust has a very flat spectral signature at typical environmental concentrations (< ~5000 ppb). The cost surface is essentially insensitive to dust concentration in this regime — the spectral effect of 100 vs 5000 ppb dust is smaller than typical measurement noise. Dust retrievals are unreliable unless:
- Concentrations are very high (> ~10,000 ppb)
- Strong prior constraints are applied via `regularization`
- Dust is fixed via `fixed_params` when it is not the primary parameter of interest

## Binary Parameters

The `direct` parameter (sky condition: 0 = diffuse, 1 = direct beam) is a binary flag. It cannot be continuously optimised — the cost surface is flat everywhere except at the two valid values, and the forward model treats it as a discrete switch.

Including `direct` in `parameters` raises a `ValueError`:

```
ValueError: Binary parameters {'direct'} cannot be continuously optimised.
Pass them via `fixed_params` instead.
```

Sky conditions are typically known from meteorological observations or assumed for a given scene, so this is not a practical limitation.

## Fixed Parameters

### Fixed per emulator (at build time)

| Parameter    | Typical value | Why fixed                                                |
| ------------ | ------------- | -------------------------------------------------------- |
| `layer_type` | 1 (solid ice) | Defines the physical model                               |
| `dz`         | [0.02, 0.146] | Weakly constrained; trades off with density              |
| `incoming`   | 0-6           | Atmospheric irradiance; discrete index, not interpolable |

To change any of these, build a separate emulator. Emulators are small (~100-200 KB), so maintaining a library for different configurations is practical.

### Fixed per retrieval (`fixed_params`)

Use `fixed_params` to constrain parameters when ancillary information is available. The parameter must be part of the emulator's input space.

```python
# Sky conditions known (direct MUST be fixed — binary parameter)
result = retrieve(..., fixed_params={"direct": 1})

# SZA, density, and dust all known
result = retrieve(
    ...,
    fixed_params={"direct": 1, "solzen": 53.2, "rho": 750, "dust": 500},
)
```

This reduces the optimisation dimensionality and tightens uncertainties on the remaining parameters.

## Log-Space Parameterisation

Impurity concentrations (`black_carbon`, `snow_algae`, `glacier_algae`, `dust`) are optimised in log10(x + 1) space internally. This is critical for robust retrieval because these parameters span orders of magnitude (0 to 500,000).

### Why log space?

In linear parameter space:
- The cost surface is extremely flat at high concentrations (the spectral signal saturates)
- Gradient-based optimisers get trapped at high concentrations where the gradient is near zero
- Initial guesses at the midpoint of bounds (e.g. 250,000 for a [0, 500,000] range) are far from typical environmental values

The log10(x + 1) transformation:
- Compresses the range: [0, 500,000] maps to [0, 5.7]
- Makes the cost surface approximately equally sensitive across orders of magnitude
- Places the initial guess near typical environmental concentrations (~700 ppb)
- The +1 offset ensures log10(0 + 1) = 0, so zero concentration maps cleanly

### User-facing behaviour

This transformation is **fully transparent** — you always provide bounds, x0, and read results in linear space. The log transformation is applied and reversed internally. Uncertainties are propagated back via the chain rule.

## Optimisation Methods

### L-BFGS-B (default) — Hybrid DE + Gradient Polish

The default method uses a two-phase strategy to combine global exploration with fast local convergence:

1. **Phase 1: Quick differential evolution (DE) pre-search.** For problems with 2+ free parameters, a quick DE run (maxiter=100, popsize=10) globally explores the parameter space to identify the basin of attraction containing the global minimum. This typically requires ~1000-3000 function evaluations.

2. **Phase 2: L-BFGS-B polishing.** The DE result (if better than the default x0) seeds a quasi-Newton L-BFGS-B optimisation that converges to machine-precision accuracy in ~100-200 additional evaluations.

**Why not pure L-BFGS-B?** The ice albedo cost surface has multiple local minima. In testing, 50 random-restart L-BFGS-B runs all converged to the same suboptimal minimum (dust overestimated by 500x). The quick DE pre-search reliably escapes local minima at modest computational cost.

**Why not pure DE?** DE alone is accurate to ~1-3 significant figures. L-BFGS-B polishing refines this to machine precision.

### Nelder-Mead

Derivative-free simplex method. More robust than L-BFGS-B when the cost surface is noisy (e.g. when using the direct forward model instead of the smooth emulator).

Bounds are enforced via a penalty function (returns 1e20 for out-of-bounds evaluations) since scipy's Nelder-Mead does not natively support box constraints in older scipy versions.

### Differential Evolution (standalone)

Rigorous global search. Use when the cost surface is known to be multimodal or when no good initial guess is available. Slower but guarantees convergence to the global minimum given sufficient iterations.

### MCMC

Full Bayesian posterior sampling via `emcee`. Reports posterior median and standard deviation. Full chains available for corner plots and correlation analysis. Requires `emcee>=3`.

### When to use each optimiser

| Method                     | Evaluations (typical)   | Best for                                                                                             |
| -------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| **L-BFGS-B** (hybrid)     | ~1,500-3,000            | Default. Fast, accurate. Hybrid DE pre-search avoids local minima.                                   |
| **Nelder-Mead**            | ~300-500                | Derivative-free fallback. Robust to noisy cost surfaces (e.g. direct forward model).                 |
| **differential_evolution** | ~500-5,000              | Global search. Use when initial guess is poor or cost surface is multimodal.                         |
| **mcmc**                   | ~50,000-200,000         | Full Bayesian posterior. Publication-quality uncertainty, parameter correlations, bimodal detection.  |

## Design Rationale

### Why upper-layer impurities only?

Impurity concentrations are retrieved for the **surface layer only** (top ~2 cm). This reflects the physics:
- Biological impurities live at the ice surface (light for photosynthesis)
- Mineral impurities accumulate via deposition and melt concentration
- The spectral signal is dominated by the surface layer
- Subsurface impurities are undetectable: photons interacting with deeply buried particles are absorbed before reaching the surface

### Why not retrieve layer thickness?

Layer thickness (`dz`) and density (`rho`) trade off nearly perfectly in their effect on optical depth: doubling thickness has the same radiative effect as doubling density. This degeneracy makes them jointly unidentifiable from spectral observations alone. Fixing `dz` breaks the degeneracy.

### Why chi-squared cost function?

Chi-squared is the standard cost for physical retrieval because:
- **Probabilistic interpretation**: negative log-likelihood under Gaussian errors
- **Uncertainty weighting**: well-measured bands contribute more
- **Free uncertainty estimates**: the Hessian at the minimum gives the covariance matrix
- **MCMC compatible**: `log_prob = -0.5 * cost`

### Why Hessian uncertainty as default?

The finite-difference Hessian at the optimum costs ~4n² extra emulator evaluations (negligible at microsecond speed). Inverting it gives the covariance matrix under a Gaussian approximation, which is accurate for well-constrained unimodal problems. Use MCMC when you need the full posterior or suspect parameter degeneracies.

## Uncertainty Estimation

### Hessian-based (default)

Applied automatically after L-BFGS-B, Nelder-Mead, and differential evolution:
1. Compute finite-difference Hessian at the optimum (4-point central differences, step size = 1e-4 × parameter range)
2. Invert to get the covariance matrix
3. Report `sqrt(diag(cov))` as 1-sigma uncertainties
4. Singular Hessian → infinity uncertainty (unconstrained parameter)

For impurity parameters (optimised in log space), the Hessian is computed in log space and uncertainties are propagated back to linear space: `sigma_linear ≈ (x + 1) × ln(10) × sigma_log`. This means uncertainty scales with the retrieved value — reflecting the physical reality that spectral sensitivity to impurities is approximately logarithmic.

### MCMC posteriors

With `method="mcmc"`, the full posterior distribution is sampled:
- Uniform prior within parameter bounds
- Log-likelihood = `-0.5 * chi_squared`
- Reports posterior median and standard deviation
- Full chains available for corner plots and correlation analysis
- Requires `emcee` (`pip install emcee>=3`)

### Regularization (Gaussian priors)

Add prior information via `regularization={name: (mean, sigma)}`. This adds a penalty term to the cost function: `((value - mean) / sigma)²`. Useful for constraining degenerate parameters when prior knowledge exists (e.g. density from in-situ measurements, dust from nearby samples).

## Worked Examples

### Field spectrometer measurement

```python
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")

# Measured 480-band albedo from ASD FieldSpec
result = retrieve(
    observed=measured_albedo,
    parameters=["rds", "rho", "black_carbon", "glacier_algae", "dust"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50},
)
print(result.summary())
```

### Sentinel-2 pixel

```python
import numpy as np

result = retrieve(
    observed=np.array([0.82, 0.78, 0.75, 0.45, 0.03]),
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    obs_uncertainty=np.array([0.02, 0.02, 0.02, 0.03, 0.05]),
    fixed_params={"direct": 1, "solzen": 50, "rho": 600, "dust": 1000},
)
```

### Partial retrieval with known density and dust

```python
result = retrieve(
    observed=measured_albedo,
    parameters=["rds", "black_carbon", "glacier_algae"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "rho": 750, "dust": 500},
)
```

### MCMC for publication-quality uncertainty

```python
result = retrieve(
    observed=measured_albedo,
    parameters=["rds", "rho", "black_carbon", "glacier_algae"],
    emulator=emu,
    method="mcmc",
    mcmc_walkers=32,
    mcmc_steps=5000,
    mcmc_burn=1000,
    fixed_params={"direct": 1, "solzen": 50},
)

# Posterior statistics
print(result.summary())

# Full chains for corner plots
flat_chain = result.chains.reshape(-1, 4)
# import corner; corner.corner(flat_chain, labels=["rds", "rho", "BC", "algae"])
```

## Known Limitations

1. **Binary parameters** — `direct` cannot be retrieved; it must be passed via `fixed_params`. This is enforced with a `ValueError`.

2. **Dust sensitivity** — mineral dust has a very flat spectral signature at typical environmental concentrations (< ~5000 ppb). Dust retrievals are unreliable unless concentrations are very high or strong priors are used. Consider fixing dust via `fixed_params` when it is not the primary parameter of interest.

3. **Band-mode information limits** — satellite band observations (4-5 broadband values) cannot constrain more than 2-3 free parameters. Fix density, sky conditions, and dust for robust band-mode retrievals.

4. **Accuracy near bounds** — MLP predictions may be less accurate at extreme edges of training range. Set bounds slightly wider than expected retrieval range.

5. **No extrapolation** — predictions outside training bounds are unreliable. The emulator clips and warns but does not refuse.

6. **Single-layer impurities** — only surface-layer impurity concentrations are retrieved. Suitable for glacier ice (surface-dominated signal) but not for distributed impurity profiles.

7. **Solid ice only** — the default emulator assumes `layer_type=1`. For snow retrievals, build a custom emulator with `layer_type=0`.

8. **No atmospheric correction** — satellite reflectances are assumed to be surface reflectance. Atmospheric correction is out of scope.

9. **Parameter degeneracies** — some combinations produce similar spectra (e.g. low-concentration BC and algae trade off; dust and algae are spectrally degenerate at broadband resolution). Hessian uncertainties will be large; MCMC reveals the correlation structure. Regularization can help.

10. **Irradiance profile fixed per emulator** — the `incoming` parameter (atmosphere) is a discrete index (0-6), not continuously interpolable. Build separate emulators for different atmospheric conditions.

## See Also

- [docs/EMULATOR.md](EMULATOR.md) — emulator architecture and design
- [docs/METHODS.md](METHODS.md) — detailed technical methods (paper-quality)
- [examples/07_inversion_spectral.py](../examples/07_inversion_spectral.py) — spectral retrieval
- [examples/08_inversion_satellite.py](../examples/08_inversion_satellite.py) — satellite band retrieval
- [examples/09_inversion_methods.py](../examples/09_inversion_methods.py) — optimiser comparison
- [examples/10_end_to_end_workflow.py](../examples/10_end_to_end_workflow.py) — full pipeline
