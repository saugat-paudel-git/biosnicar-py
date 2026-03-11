#!/usr/bin/env python3
"""Quick-start example for BioSNICAR.

Run with:  python main.py

For more examples, see the examples/ directory.
"""

from biosnicar import run_model
from biosnicar.drivers.sweep import parameter_sweep

# --- Forward model with overrides ---
outputs = run_model(solzen=50, rds=1000, black_carbon=5000)
print(f"Broadband albedo: {outputs.BBA:.4f}")

# Chain onto satellite bands
s2 = outputs.to_platform("sentinel2")
print(f"Sentinel-2 B3: {s2.B3:.4f}, NDSI: {s2.NDSI:.4f}")

# --- Parameter sweep ---
df = parameter_sweep(
    params={"solzen": [30, 50, 70], "rds": [200, 1000, 5000]},
    progress=False,
)
print("\nSweep results:")
print(df[["solzen", "rds", "BBA"]].to_string(index=False))
