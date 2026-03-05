#!/usr/bin/env python3
"""End-to-end workflow: emulator -> synthetic observation -> retrieval -> validation.

Demonstrates the full pipeline: load (or build) an emulator, generate a
synthetic Sentinel-2 observation, retrieve ice properties, validate against
truth, and convolve the result to platform bands.
"""

import numpy as np

from biosnicar import run_emulator, to_platform
from biosnicar.emulator import Emulator
from biosnicar.inverse import retrieve

PLOT = False

# ======================================================================
# Step 1: Load pre-built emulator
# ======================================================================
print("Step 1: Load emulator\n")
emu = Emulator.load("data/emulators/glacier_ice_7_param_default.npz")
print(f"  {emu!r}")

# ======================================================================
# Step 2: Define a "true" ice surface and generate synthetic observation
# ======================================================================
print("\nStep 2: Generate synthetic Sentinel-2 observation\n")
true_params = {
    "rds": 1200,  # bubble radius (um)
    "rho": 700,  # density (kg/m3)
    "black_carbon": 3000,  # black carbon (ppb)
    "glacier_algae": 80000,  # glacier algae (cells/mL)
}

# Generate full spectrum via emulator
true_albedo = emu.predict(**true_params)

# Convolve to Sentinel-2 bands
s2_true = to_platform(true_albedo, "sentinel2", flx_slr=emu.flx_slr)

# Select bands to use for retrieval
band_names = ["B2", "B3", "B4", "B8", "B11"]
obs_values = np.array([getattr(s2_true, b) for b in band_names])

# Add realistic measurement noise
rng = np.random.RandomState(42)
noise_sigma = np.array([0.02, 0.02, 0.02, 0.03, 0.05])
obs_noisy = obs_values + rng.normal(0, noise_sigma)
obs_noisy = np.clip(obs_noisy, 0, 1)

print("  True parameters:")
for k, v in true_params.items():
    print(f"    {k}: {v}")
print(f"\n  S2 bands (noisy): {dict(zip(band_names, obs_noisy.round(4)))}")

# ======================================================================
# Step 3: Retrieve ice properties
# ======================================================================
print("\nStep 3: Retrieve ice properties\n")
result = retrieve(
    observed=obs_noisy,
    parameters=["rds", "rho", "black_carbon", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=band_names,
    obs_uncertainty=noise_sigma,
)
print(result.summary())

# ======================================================================
# Step 4: Validate against truth
# ======================================================================
print("\n\nStep 4: Validation\n")
print(
    f"  {'Parameter':25s} {'True':>10s} {'Retrieved':>10s} {'Error':>10s} {'Unc':>10s}"
)
print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
for name in result.best_fit:
    true_val = true_params[name]
    ret_val = result.best_fit[name]
    err = abs(ret_val - true_val)
    unc = result.uncertainty[name]
    print(f"  {name:25s} {true_val:10.0f} {ret_val:10.1f} {err:10.1f} {unc:10.1f}")

# ======================================================================
# Step 5: Use retrieved parameters downstream
# ======================================================================
print("\n\nStep 5: Use retrieved spectrum downstream\n")

# Generate outputs from retrieved parameters via run_emulator()
outputs = run_emulator(emu, **result.best_fit)
print(f"  BBA (retrieved):     {outputs.BBA:.4f}")

# Compute true BBA for comparison
true_bba = float(np.sum(emu.flx_slr * true_albedo) / np.sum(emu.flx_slr))
print(f"  BBA (true):          {true_bba:.4f}")

# Convolve retrieved spectrum to other platforms
s2_ret = outputs.to_platform("sentinel2")
cesm_ret = outputs.to_platform("cesm2band")
print(f"  S2 NDSI (retrieved): {s2_ret.NDSI:.4f}")
print(f"  CESM vis (retr.):    {cesm_ret.vis:.4f}")
print(f"  CESM nir (retr.):    {cesm_ret.nir:.4f}")

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: spectral comparison
    ax1.plot(wavelengths, true_albedo, "k-", label="True", linewidth=1.5)
    ax1.plot(wavelengths, result.predicted_albedo, "r--", label="Retrieved")
    ax1.set_xlabel("Wavelength (um)")
    ax1.set_ylabel("Albedo")
    ax1.set_xlim(0.2, 2.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("True vs retrieved spectrum")
    ax1.legend()

    # Right: parameter comparison bar chart
    names = list(true_params.keys())
    true_vals = [true_params[n] for n in names]
    ret_vals = [result.best_fit[n] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax2.bar(x - width / 2, true_vals, width, label="True")
    ax2.bar(x + width / 2, ret_vals, width, label="Retrieved")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Value")
    ax2.set_title("Parameter recovery")
    ax2.legend()

    fig.tight_layout()
    plt.show()
