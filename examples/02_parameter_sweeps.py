#!/usr/bin/env python3
"""Parameter sweep examples.

Demonstrates ``parameter_sweep()`` for sensitivity analysis: 2-parameter,
3-parameter, impurity concentration, and spectral output sweeps.
"""

import numpy as np

from biosnicar.drivers.sweep import parameter_sweep

PLOT = True

# ======================================================================
# Example 1: Two-parameter sweep (SZA x grain radius)
# ======================================================================
print("=== Example 1: SZA x grain radius (20 combinations) ===\n")
df = parameter_sweep(
    params={
        "solzen": [30, 40, 50, 60, 70],
        "rds": [100, 200, 500, 1000],
    },
)

pivot = df.pivot_table(values="BBA", index="solzen", columns="rds")
print("Broadband albedo pivot table:\n")
print(pivot.to_string(float_format="{:.4f}".format))

# ======================================================================
# Example 2: Impurity sweep (black carbon)
# ======================================================================
print("\n\n=== Example 2: Black carbon concentration sweep ===\n")
df_bc = parameter_sweep(
    params={"black_carbon": [0, 100, 1000, 10000, 100000]},
)

for _, row in df_bc.iterrows():
    print(f"  BC = {row['black_carbon']:>8.0f} ppb  ->  BBA = {row['BBA']:.4f}")

# ======================================================================
# Example 3: Three-parameter sweep
# ======================================================================
print("\n\n=== Example 3: SZA x density x grain radius (36 combos) ===\n")
df3 = parameter_sweep(
    params={
        "solzen": [40, 50, 60],
        "rho": [400, 600, 800],
        "rds": [200, 500, 1000, 5000],
    },
)

for sza in sorted(df3["solzen"].unique()):
    subset = df3[df3["solzen"] == sza]
    piv = subset.pivot_table(values="BBA", index="rho", columns="rds")
    print(f"  SZA = {sza} deg:\n")
    print("  " + piv.to_string(float_format="{:.4f}".format).replace("\n", "\n  "))
    print()

# ======================================================================
# Example 4: Spectral output
# ======================================================================
print("\n=== Example 4: Spectral output for three grain radii ===\n")
df_spec = parameter_sweep(
        params={
        "solzen": [40, 50, 60],
        "rho": [400, 600, 800],
        "rds": [200, 500, 1000, 5000],
        "glacier_algae": [1000, 5000, 10000, 50000, 100000],
        "solzen": [20, 30, 40, 60, 70]
    },
    return_spectral=True,
    progress=False,
)

for _, row in df_spec.iterrows():
    alb = row["albedo"]
    print(
        f"  rds = {row['rds']:>5.0f} um  ->  BBA = {row['BBA']:.4f}  "
        f"(spectral range [{alb.min():.4f}, {alb.max():.4f}])"
    )

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)

    # Sweep plot
    fig, ax = plt.subplots()
    for rds in pivot.columns:
        ax.plot(pivot.index, pivot[rds], marker="o", label=f"r = {rds} um")
    ax.set_xlabel("Solar zenith angle (deg)")
    ax.set_ylabel("Broadband albedo")
    ax.set_title("BBA vs SZA for different grain radii")
    ax.legend()
    fig.tight_layout()

    # Spectral plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in df_spec.iterrows():
        ax.plot(wavelengths, row["albedo"], label=f"r = {int(row['rds'])} um")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Albedo")
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(-0.10, 1.1)
    ax.set_title("Spectral albedo for different grain radii")
    ax.legend()
    fig.tight_layout()
    plt.show()
