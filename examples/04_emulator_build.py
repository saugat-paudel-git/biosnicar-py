#!/usr/bin/env python3
"""Building a custom emulator.

Demonstrates ``Emulator.build()`` with a small training set, inspecting
emulator properties, and validating accuracy against the forward model.

Note: this example builds a small emulator (200 samples) for speed.
Production emulators should use 5000-10000+ samples.
"""

import time

import numpy as np

from biosnicar import run_model
from biosnicar.emulator import Emulator

PLOT = False

# ======================================================================
# Example 1: Build a 2-parameter emulator
# ======================================================================
print("=== Example 1: Build a 2-parameter emulator (200 samples) ===\n")
t0 = time.time()
emu = Emulator.build(
    params={
        "rds": (100, 5000),
        "black_carbon": (0, 50000),
    },
    n_samples=200,
    layer_type=1,  # fixed: solid ice
    solzen=50,  # fixed: SZA
    direct=1,  # fixed: clear sky
    progress=True,
)
build_time = time.time() - t0
print(f"\n  Build time:       {build_time:.1f} s")

# ======================================================================
# Example 2: Inspect emulator properties
# ======================================================================
print("\n=== Example 2: Emulator properties ===\n")
print(f"  Parameters:       {emu.param_names}")
print(f"  Bounds:           {emu.bounds}")
print(f"  PCA components:   {emu.n_pca_components}")
print(f"  Training R2:      {emu.training_score:.6f}")
print(f"  repr:             {emu!r}")

# ======================================================================
# Example 3: Validate accuracy against run_model()
# ======================================================================
print("\n=== Example 3: Accuracy validation ===\n")
test_points = [
    {"rds": 500, "black_carbon": 0},
    {"rds": 1000, "black_carbon": 5000},
    {"rds": 2000, "black_carbon": 20000},
    {"rds": 5000, "black_carbon": 50000},
]

errors = []
for params in test_points:
    emu_albedo = emu.predict(**params)
    ref_outputs = run_model(solzen=50, direct=1, layer_type=1, **params)
    ref_albedo = np.array(ref_outputs.albedo)

    mae = np.mean(np.abs(emu_albedo - ref_albedo))
    max_err = np.max(np.abs(emu_albedo - ref_albedo))
    errors.append(mae)

    print(
        f"  rds={params['rds']:>5d}, BC={params['black_carbon']:>6d}  "
        f"MAE={mae:.6f}  MaxErr={max_err:.6f}"
    )

print(f"\n  Overall mean MAE: {np.mean(errors):.6f}")

# ======================================================================
# Example 4: Build with more parameters
# ======================================================================
print("\n=== Example 4: 4-parameter emulator (300 samples) ===\n")
emu4 = Emulator.build(
    params={
        "rds": (100, 5000),
        "rho": (100, 917),
        "black_carbon": (0, 100000),
        "glacier_algae": (0, 500000),
    },
    n_samples=300,
    layer_type=1,
    solzen=50,
    direct=1,
    progress=True,
)
print(f"\n  Parameters:       {emu4.param_names}")
print(f"  PCA components:   {emu4.n_pca_components}")
print(f"  Training R2:      {emu4.training_score:.6f}")

if PLOT:
    import matplotlib.pyplot as plt

    wavelengths = np.arange(0.205, 4.999, 0.01)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, params in zip(axes.flat, test_points):
        emu_alb = emu.predict(**params)
        ref = run_model(solzen=50, direct=1, layer_type=1, **params)
        ref_alb = np.array(ref.albedo)

        ax.plot(wavelengths, ref_alb, "k-", label="Forward model", linewidth=1.5)
        ax.plot(wavelengths, emu_alb, "r--", label="Emulator", linewidth=1)
        ax.set_xlim(0.2, 2.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"rds={params['rds']}, BC={params['black_carbon']}")
        ax.legend(fontsize=8)

    fig.suptitle("Emulator vs forward model (200 training samples)")
    fig.tight_layout()
    plt.show()
