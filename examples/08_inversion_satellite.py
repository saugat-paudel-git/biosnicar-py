#!/usr/bin/env python3
"""Satellite band-mode retrieval.

Demonstrates ``retrieve()`` with ``platform="sentinel2"`` (and other
platforms), ``obs_uncertainty``, and working with band-space observations.
"""

import numpy as np

from biosnicar import to_platform
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

PLOT = True

# Load emulator
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")

# Generate synthetic Sentinel-2 observation from known parameters
true_params = dict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000, dust=1000, solzen=50, direct=1)
true_albedo = emu.predict(**true_params)
s2_obs = to_platform(true_albedo, "sentinel2", flx_slr=emu.flx_slr)

# Use a subset of bands for retrieval
band_names = ["B2", "B3", "B4", "B8", "B11"]
obs_values = np.array([getattr(s2_obs, b) for b in band_names])
print("Synthetic Sentinel-2 observation:")
for name, val in zip(band_names, obs_values):
    print(f"  {name}: {val:.4f}")

# ======================================================================
# Example 1: Basic Sentinel-2 retrieval
# ======================================================================
print("\n=== Example 1: Sentinel-2 retrieval ===\n")
result = retrieve(
    observed=obs_values,
    parameters=["rds", "rho", "black_carbon", "glacier_algae", "dust", "solzen", "direct"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
)
print(result.summary())
print(f"\n  True:      {true_params}")

# ======================================================================
# Example 2: With measurement uncertainty
# ======================================================================
print("\n\n=== Example 2: With measurement uncertainty ===\n")
obs_unc = np.array([0.02, 0.02, 0.02, 0.03, 0.05])  # per-band 1-sigma

result2 = retrieve(
    observed=obs_values,
    parameters=["rds", "rho", "black_carbon", "glacier_algae",  "dust", "solzen", "direct"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    obs_uncertainty=obs_unc,
)
for name in result2.best_fit:
    print(
        f"  {name:25s} = {result2.best_fit[name]:10.1f} +/- {result2.uncertainty[name]:.1f}"
    )

# ======================================================================
# Example 3: Partial retrieval (fix density)
# ======================================================================
print("\n\n=== Example 3: Fix density, retrieve impurities ===\n")
result3 = retrieve(
    observed=obs_values,
    parameters=["rho", "rds", "black_carbon", "glacier_algae",  "dust"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    fixed_params={"solzen":50, "direct":1 },
)
for name in result3.best_fit:
    print(
        f"  {name:25s} = {result3.best_fit[name]:10.1f} (true: {true_params[name]:.0f})"
    )

# ======================================================================
# Example 4: Landsat 8 retrieval
# ======================================================================
print("\n\n=== Example 4: Landsat 8 retrieval ===\n")
l8_obs = to_platform(true_albedo, "landsat8", flx_slr=emu.flx_slr)
l8_band_names = ["B2", "B3", "B4", "B5", "B6"]
l8_values = np.array([getattr(l8_obs, b) for b in l8_band_names])

result4 = retrieve(
    observed=l8_values,
    parameters=["rds", "rho", "black_carbon", "glacier_algae", "dust", "solzen", "direct"],
    emulator=emu,
    platform="landsat8",
    observed_band_names=l8_band_names,
)
print(f"  Landsat 8 retrieved rds: {result4.best_fit['rds']:.0f} (true: 1000)")
print(f"  Converged: {result4.converged}")

# ======================================================================
# Example 5: MODIS retrieval
# ======================================================================
print("\n\n=== Example 5: MODIS retrieval ===\n")
modis_obs = to_platform(true_albedo, "modis", flx_slr=emu.flx_slr)
modis_band_names = ["B1", "B2", "B3", "B4"]
modis_values = np.array([getattr(modis_obs, b) for b in modis_band_names])

result5 = retrieve(
    observed=modis_values,
    parameters=["rds", "rho", "black_carbon", "glacier_algae", "dust", "solzen", "direct"],
    emulator=emu,
    platform="modis",
    observed_band_names=modis_band_names,
)
print(f"  MODIS retrieved rds: {result5.best_fit['rds']:.0f} (true: 1000)")

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
    ax.set_title("Band-mode retrieval from S2 and L8 observations")
    ax.legend()
    fig.tight_layout()
    plt.show()
