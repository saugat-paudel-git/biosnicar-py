#!/usr/bin/env python3
"""Satellite band-mode retrieval.

This script demonstrates how to retrieve ice physical properties from
satellite band observations (e.g. Sentinel-2, Landsat 8, MODIS) using
BioSNICAR's ``retrieve()`` function with the ``platform`` keyword.

In band mode, the emulator predicts the full 480-band spectrum internally,
convolves it to the satellite's spectral response functions, and compares
to your broadband observations.  You never need to reconstruct a
continuous spectrum from satellite data.

IMPORTANT: Band-mode retrieval provides substantially less spectral
information than full spectral mode (typically 4-5 broadband values vs 480
spectral bands).  This limits the number of parameters that can be
reliably constrained.  Best practice is to:
  - Fix poorly-constrained parameters (dust, sky conditions) via fixed_params
  - Limit free parameters to 2-3 (typically SSA plus 1-2 impurities)
  - Use obs_uncertainty to weight bands according to measurement quality

SSA is the recommended retrieval metric because it collapses the rds/rho
degeneracy into a single well-constrained parameter — particularly
important when observations are limited to a few broadband values.
"""

import numpy as np

from biosnicar import run_model, to_platform
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve
from biosnicar.inverse.result import _compute_ssa

PLOT = True

# ======================================================================
# Setup: load emulator and generate a synthetic satellite observation
# ======================================================================

# Load the pre-built 8-parameter glacier ice emulator.
emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")

# Parameters that are known a priori and will NOT be retrieved.
# Dust is fixed at 1000 ppb because it has very low spectral sensitivity
# at typical environmental concentrations — the effect of 100 vs 5000 ppb
# is smaller than typical measurement noise (see docs/INVERSION.md).
fixed = {"solzen": 50, "direct": 1, "dust": 1000, "snow_algae": 0}

# Define the "true" ice surface.  We generate the observation from the
# FULL FORWARD MODEL (not the emulator) so that the retrieval is honest.
# In real use, the observation comes from a satellite image.
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)
true_outputs = run_model(**true_params, **fixed, layer_type=1)
true_albedo = np.array(true_outputs.albedo, dtype=np.float64)

# Convolve the full spectrum to Sentinel-2 bands using the emulator's
# solar flux spectrum for flux weighting.  This is what a Sentinel-2
# pixel would measure over this ice surface.
s2_obs = to_platform(true_albedo, "sentinel2", flx_slr=emu.flx_slr)

# Compute true SSA for validation.
true_ssa = _compute_ssa(true_params["rds"], true_params["rho"])
print(f"True SSA: {true_ssa:.4f} m2/kg  (from rds={true_params['rds']}, rho={true_params['rho']})\n")

# SSA + impurities to retrieve.  SSA replaces separate rds/rho retrieval,
# which is especially important in band mode where the limited spectral
# information cannot resolve the rds/rho degeneracy.
ice_params = ["ssa", "black_carbon", "glacier_algae"]

# Select which Sentinel-2 bands to use.  These span visible through SWIR
# and capture both impurity absorption (VIS) and ice grain scattering (NIR/SWIR).
band_names = ["B2", "B3", "B4", "B8", "B11"]
obs_values = np.array([getattr(s2_obs, b) for b in band_names])
print("Synthetic Sentinel-2 observation:")
for name, val in zip(band_names, obs_values):
    print(f"  {name}: {val:.4f}")

# ======================================================================
# Example 1: Basic Sentinel-2 retrieval
# ======================================================================
# The simplest band-mode workflow.  Pass the band values, the platform
# name, and the band names.  The emulator handles the internal
# full-spectrum prediction and band convolution automatically.

print("\n=== Example 1: Sentinel-2 SSA retrieval ===\n")
result = retrieve(
    observed=obs_values,
    parameters=ice_params,
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
)
print(result.summary())
print(f"\n  True SSA: {true_ssa:.4f},  Retrieved SSA: {result.best_fit['ssa']:.4f}")
print(f"  Internal decomposition: {result.derived}")

# ======================================================================
# Example 2: With measurement uncertainty
# ======================================================================
# Real satellite observations have per-band uncertainty from calibration,
# atmospheric correction, and sensor noise.  Passing obs_uncertainty
# enables chi-squared weighting: well-measured bands (low sigma) contribute
# more to the cost function.
#
# Typical per-band uncertainties for Sentinel-2 L2A surface reflectance:
# - VIS bands (B2-B4): ~0.02 (good calibration)
# - NIR (B8): ~0.03 (moderate)
# - SWIR (B11): ~0.05 (higher uncertainty from atmospheric water vapour)

print("\n\n=== Example 2: With measurement uncertainty ===\n")
obs_unc = np.array([0.02, 0.02, 0.02, 0.03, 0.05])  # per-band 1-sigma

result2 = retrieve(
    observed=obs_values,
    parameters=ice_params,
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
    obs_uncertainty=obs_unc,
)
for name in result2.best_fit:
    print(f"  {name:25s} = {result2.best_fit[name]:10.4f}")

# ======================================================================
# Example 3: Fewer free parameters
# ======================================================================
# With only 5 broadband observations, information content is limited.
# Fixing additional parameters (here the impurity concentrations are
# already minimal) can improve the retrieval of the remaining ones.
# This example shows that even with 3 free parameters, band mode can
# produce reasonable results for SSA and dominant impurities.

print("\n\n=== Example 3: Retrieve SSA + impurities (dust fixed) ===\n")
result3 = retrieve(
    observed=obs_values,
    parameters=["ssa", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params=fixed,
)
print(f"  SSA:            {result3.best_fit['ssa']:10.4f}  (true: {true_ssa:.4f})")
print(f"  black_carbon:   {result3.best_fit['black_carbon']:10.1f}  (true: {true_params['black_carbon']:.0f})")
print(f"  glacier_algae:  {result3.best_fit['glacier_algae']:10.1f}  (true: {true_params['glacier_algae']:.0f})")

# ======================================================================
# Example 4: Landsat 8 retrieval
# ======================================================================
# The same workflow applies to any supported platform.  Landsat 8 has
# fewer spectral bands than Sentinel-2 but covers a similar wavelength
# range.  Note that different platforms have different band names (B2-B6
# for Landsat 8 OLI vs B2-B11 for Sentinel-2).

print("\n\n=== Example 4: Landsat 8 SSA retrieval ===\n")
l8_obs = to_platform(true_albedo, "landsat8", flx_slr=emu.flx_slr)
l8_band_names = ["B2", "B3", "B4", "B5", "B6"]
l8_values = np.array([getattr(l8_obs, b) for b in l8_band_names])

result4 = retrieve(
    observed=l8_values,
    parameters=ice_params,
    emulator=emu,
    platform="landsat8",
    observed_band_names=l8_band_names,
    fixed_params=fixed,
)
print(f"  Landsat 8 retrieved SSA: {result4.best_fit['ssa']:.4f} (true: {true_ssa:.4f})")
print(f"  Converged: {result4.converged}")

# ======================================================================
# Example 5: MODIS retrieval
# ======================================================================
# MODIS provides daily global coverage but has broader spectral bands
# and lower spatial resolution than Sentinel-2 or Landsat 8.  The same
# retrieval framework applies — the emulator handles the SRF convolution
# differences automatically.

print("\n\n=== Example 5: MODIS SSA retrieval ===\n")
modis_obs = to_platform(true_albedo, "modis", flx_slr=emu.flx_slr)
modis_band_names = ["B1", "B2", "B3", "B4"]
modis_values = np.array([getattr(modis_obs, b) for b in modis_band_names])

result5 = retrieve(
    observed=modis_values,
    parameters=ice_params,
    emulator=emu,
    platform="modis",
    observed_band_names=modis_band_names,
    fixed_params=fixed,
)
print(f"  MODIS retrieved SSA: {result5.best_fit['ssa']:.4f} (true: {true_ssa:.4f})")

# ======================================================================
# Optional: plot true spectrum vs retrieved spectra from different sensors
# ======================================================================
if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavelengths, true_albedo, "k-", label="True spectrum", alpha=0.5)
    ax.plot(wavelengths, result.predicted_albedo, "r--", label="S2 retrieval")
    ax.plot(wavelengths, result4.predicted_albedo, "b:", label="L8 retrieval")
    for name, val in zip(band_names, obs_values):
        ax.axhline(val, color="green", alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("Band-mode SSA retrieval from S2 and L8 observations")
    ax.legend()
    fig.tight_layout()
    plt.show()
