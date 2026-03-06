#!/usr/bin/env python3
"""Spectral retrieval with the inverse module.

Demonstrates ``retrieve()`` in spectral mode (480-band observed albedo),
including ``fixed_params``, ``wavelength_mask``, ``regularization``, and
result inspection.
"""

import numpy as np

from biosnicar import run_model
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

PLOT = True

# Load the default emulator
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")
print(f"Emulator: {emu!r}\n")

# Observing conditions (known, not retrieved)
fixed = {"solzen": 50, "direct": 1}

# Generate a synthetic observation from the FULL FORWARD MODEL (not the
# emulator) so that the retrieval is honest — the emulator must bridge the
# approximation gap between its MLP predictions and the true RT solver output.
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=100000, dust=1000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
observed = np.array(true_outputs.albedo, dtype=np.float64)

# Ice properties to retrieve
ice_params = ["rds", "rho", "black_carbon", "dust", "glacier_algae"]

# ======================================================================
# Example 1: Basic spectral retrieval
# ======================================================================
print("=== Example 1: Basic spectral retrieval ===\n")
result = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
)
print(result.summary())
print(f"\n  True:      {true_params}")
print(f"  Retrieved: {result.best_fit}")

# ======================================================================
# Example 2: Partial retrieval with fixed_params
# ======================================================================
print("\n\n=== Example 2: Fix density, retrieve the rest ===\n")
result2 = retrieve(
    observed=observed,
    parameters=["rds", "black_carbon", "dust", "glacier_algae"],
    emulator=emu,
    fixed_params={**fixed, "rho": 600},  # density known from in-situ measurement
)
print(result2.summary())

# ======================================================================
# Example 3: Wavelength mask (exclude noisy regions)
# ======================================================================
print("\n\n=== Example 3: Wavelength mask ===\n")
wavelengths = np.arange(0.205, 4.999, 0.01)
# Use only visible-NIR (0.3-2.5 um), exclude thermal
mask = (wavelengths >= 0.3) & (wavelengths <= 2.5)
print(f"  Using {mask.sum()} of 480 bands (0.3-2.5 um)")

result3 = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    wavelength_mask=mask,
)
print(f"  Retrieved rds: {result3.best_fit['rds']:.0f} (true: 1000)")

# ======================================================================
# Example 4: Observation uncertainty
# ======================================================================
print("\n\n=== Example 4: Observation uncertainty weighting ===\n")
# Simulate noisy observation
rng = np.random.RandomState(42)
noise_sigma = 0.02
noisy_obs = observed + rng.normal(0, noise_sigma, size=observed.shape)
noisy_obs = np.clip(noisy_obs, 0, 1)

# With uncertainty weighting
obs_unc = np.full(480, noise_sigma)
result4 = retrieve(
    observed=noisy_obs,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    obs_uncertainty=obs_unc,
)
print(f"  Retrieved with uncertainty weighting:")
for name in result4.best_fit:
    print(
        f"    {name:25s} = {result4.best_fit[name]:12.1f} +/- {result4.uncertainty[name]:.1f}"
    )

# ======================================================================
# Example 5: Regularization (Gaussian priors)
# ======================================================================
print("\n\n=== Example 5: Regularization with priors ===\n")
result5 = retrieve(
    observed=observed,
    parameters=ice_params,
    emulator=emu,
    fixed_params=fixed,
    regularization={
        "rho": (600, 50),  # prior: 600 +/- 50 kg/m3
        "black_carbon": (3000, 5000),  # prior: 3000 +/- 5000 ppb
    },
)
print(f"  With regularization:")
for name in result5.best_fit:
    print(f"    {name:25s} = {result5.best_fit[name]:12.1f}")

# ======================================================================
# Example 6: Inspect result object
# ======================================================================
print("\n\n=== Example 6: Result object inspection ===\n")
print(f"  converged:       {result.converged}")
print(f"  method:          {result.method}")
print(f"  cost:            {result.cost:.6f}")
print(f"  n_function_evals: {result.n_function_evals}")
print(f"  predicted shape: {result.predicted_albedo.shape}")
print(f"  observed shape:  {result.observed.shape}")

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, observed, "k-", label="Observed", linewidth=1.5)
    ax.plot(wavelengths, result.predicted_albedo, "r--", label="Retrieved", linewidth=1)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spectral retrieval: observed vs retrieved")
    ax.legend()
    fig.tight_layout()
    plt.show()
