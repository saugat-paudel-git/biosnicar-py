#!/usr/bin/env python3
"""Build the default glacier-ice emulator shipped with the repository.

Creates a 4-parameter emulator (rds, rho, black_carbon, glacier_algae)
trained on 5000 Latin hypercube samples and saves it to
``data/emulators/glacier_ice_7_param_default.npz``.

Usage:
    python scripts/build_default_emulator.py
"""

from pathlib import Path

from biosnicar.emulator import Emulator

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "emulators"
    / "glacier_ice_7_param_default.npz"
)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Building default glacier-ice emulator (150000 samples)...")
    emu = Emulator.build(
        params={  # provide min and max for each param, then n_samples.
            "rds": (100, 10000),
            "rho": (100, 900),
            "black_carbon": (0, 5000),
            "dust": (0, 500000),
            "glacier_algae": (0, 500000),
            "direct": (0, 1),
            "solzen": (25, 65)
        },
        n_samples=150000,
        layer_type=1,
        direct=1,
        progress=True,
    )

    emu.save(OUTPUT)
    print(f"\nSaved to {OUTPUT}")
    print(f"  Parameters:     {emu.param_names}")
    print(f"  PCA components: {emu.n_pca_components}")
    print(f"  Training R²:    {emu.training_score:.6f}")


if __name__ == "__main__":
    main()
