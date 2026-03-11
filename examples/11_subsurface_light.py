#!/usr/bin/env python3
"""Subsurface light field: fluxes, PAR depth profiles, spectral heating.

Demonstrates ``outputs.F_up``, ``outputs.F_dwn``, ``subsurface_flux()``,
``par()``, and ``spectral_heating_rate()``.
"""

import sys

import numpy as np

from biosnicar import run_model

PLOT = "--plot" in sys.argv

# ======================================================================
# Example 1: Basic subsurface flux arrays
# ======================================================================
print("=== Example 1: Subsurface flux arrays ===\n")
outputs = run_model(
    solzen=60, rds=500,
    dz=[0.01, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5],
    rho=[400] * 10,
)
print(f"  F_up  shape: {outputs.F_up.shape}  (nbr_wvl, nbr_lyr+1)")
print(f"  F_dwn shape: {outputs.F_dwn.shape}")
print(f"  Surface F_dwn sum (total incoming): {np.sum(outputs.F_dwn[:, 0]):.4f}")
print(f"  Bottom  F_dwn sum:                  {np.sum(outputs.F_dwn[:, -1]):.4f}")

flux_5cm = outputs.subsurface_flux(0.05)
print(f"  F_dwn at 5 cm shape: {flux_5cm['F_dwn'].shape}")

# ======================================================================
# Example 2: PAR depth profile
# ======================================================================
print("\n=== Example 2: PAR depth profile ===\n")
depths = np.linspace(0, 0.5, 11)
par_values = outputs.par(depths)

print(f"  {'Depth (m)':>10s}  {'PAR':>10s}")
print(f"  {'-' * 10}  {'-' * 10}")
for d, p in zip(depths, par_values):
    print(f"  {d:10.3f}  {p:10.4f}")

# ======================================================================
# Example 3: Clean vs dirty ice PAR comparison
# ======================================================================
print("\n=== Example 3: Clean vs impurity-loaded PAR ===\n")
_dz_profile = [0.01, 0.02, 0.02, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
_rho_profile = [400] * len(_dz_profile)
clean = run_model(solzen=50, rds=500, dz=_dz_profile, rho=_rho_profile)
bc = run_model(solzen=50, rds=500, dz=_dz_profile, rho=_rho_profile, black_carbon=5000)
algae = run_model(solzen=50, rds=500, dz=_dz_profile, rho=_rho_profile, glacier_algae=50000)

depths = np.array([0.0, 0.01, 0.05, 0.1, 0.25, 0.5])
par_clean = clean.par(depths)
par_bc = bc.par(depths)
par_algae = algae.par(depths)

print(f"  {'Depth':>8s}  {'Clean':>8s}  {'BC':>8s}  {'Algae':>8s}")
print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
for d, pc, pb, pa in zip(depths, par_clean, par_bc, par_algae):
    print(f"  {d:8.3f}  {pc:8.4f}  {pb:8.4f}  {pa:8.4f}")

# ======================================================================
# Example 4: Spectral flux at selected depths
# ======================================================================
print("\n=== Example 4: Spectral flux at depth ===\n")
wvl = outputs._wavelengths
for depth in [0.0, 0.05, 0.2]:
    flux = outputs.subsurface_flux(depth)
    vis_sum = np.sum(flux["F_dwn"][(wvl >= 0.4) & (wvl <= 0.7)])
    nir_sum = np.sum(flux["F_dwn"][(wvl >= 0.7) & (wvl <= 1.3)])
    print(f"  depth={depth:.2f} m  VIS F_dwn sum={vis_sum:.4f}  NIR F_dwn sum={nir_sum:.4f}")

# ======================================================================
# Example 5: Spectral heating rate
# ======================================================================
print("\n=== Example 5: Spectral heating rate ===\n")
shr = outputs.spectral_heating_rate()
print(f"  Shape: {shr.shape}  (nbr_wvl, nbr_lyr)")
broadband_hr = np.sum(shr, axis=0)
print(f"  Broadband heating rate per layer (K/hr): {broadband_hr}")

# Peak wavelength per layer
for lyr in range(shr.shape[1]):
    peak_idx = np.argmax(np.abs(shr[:, lyr]))
    print(f"  Layer {lyr}: peak heating at {wvl[peak_idx]:.2f} um "
          f"({shr[peak_idx, lyr]:.4f} K/hr)")

# ======================================================================
# Optional plots
# ======================================================================
if PLOT:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Spectral F_dwn at several depths
    ax = axes[0, 0]
    for depth in [0.0, 0.02, 0.05, 0.2]:
        flux = outputs.subsurface_flux(depth)
        ax.plot(wvl, flux["F_dwn"], label=f"{depth:.2f} m")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Downwelling flux (normalised)")
    ax.set_xlim(0.3, 2.5)
    ax.set_title("Spectral downwelling flux at depth")
    ax.legend()

    # Panel 2: PAR vs depth for clean/dirty
    ax = axes[0, 1]
    depths_fine = np.linspace(0, 0.5, 50)
    ax.plot(depths_fine, clean.par(depths_fine), label="Clean")
    ax.plot(depths_fine, bc.par(depths_fine), label="BC 5000 ppb")
    ax.plot(depths_fine, algae.par(depths_fine), label="Algae 50k cells/mL")
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("PAR (normalised)")
    ax.set_title("PAR depth profile")
    ax.legend()

    # Panel 3: Spectral heating rate heatmap
    ax = axes[1, 0]
    vis_mask = (wvl >= 0.3) & (wvl <= 2.5)
    wvl_vis = wvl[vis_mask]
    im = ax.imshow(
        shr[vis_mask, :],
        aspect="auto",
        origin="lower",
        extent=[0, shr.shape[1], wvl_vis[0], wvl_vis[-1]],
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Wavelength (um)")
    ax.set_title("Spectral heating rate (K/hr)")
    fig.colorbar(im, ax=ax)

    # Panel 4: Broadband flux profile through column
    ax = axes[1, 1]
    z_interfaces = np.concatenate(([0], np.cumsum(outputs._dz)))
    bb_up = np.sum(outputs.F_up, axis=0)
    bb_dwn = np.sum(outputs.F_dwn, axis=0)
    ax.plot(bb_dwn, z_interfaces, "b-o", label="F_dwn (broadband)")
    ax.plot(bb_up, z_interfaces, "r-o", label="F_up (broadband)")
    ax.invert_yaxis()
    ax.set_xlabel("Broadband flux (normalised)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Flux profile through column")
    ax.legend()

    fig.tight_layout()
    plt.show()
