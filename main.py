#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from biosnicar import run_model
from biosnicar.drivers.sweep import parameter_sweep
from biosnicar.bands import to_platform

outputs = run_model(
    solver="adding-doubling",
    plot=False,
    validate=True,
    solzen=50,
    layer_type=[1, 1, 1, 1, 1],
    dz=[0.02, 0.05, 0.05, 0.05, 0.83],
    rds=[800, 900, 1000, 1100, 1200],
    rho=[500, 600, 700, 750, 800],
    lwc=[0, 0, 0, 0, 0],
    impurity_0_conc=[0, 0, 0, 0, 0],       # black carbon (ppb)
    impurity_1_conc=[40000, 10000, 0, 0, 0],  # snow algae (cells/mL)
    impurity_2_conc=[0, 0, 0, 0, 0],       # glacier algae (cells/mL)
)

df = parameter_sweep(
    params={
        "solzen": [30, 40, 50, 60, 70],
        "rds": [100, 200, 500, 1000],
    }
)

# 1. Run the forward model
outputs = run_model(solzen=50, rds=1000)

# 2. Convolve onto any platform
s2  = to_platform(outputs.albedo, "sentinel2",  flx_slr=outputs.flx_slr)
cesm = to_platform(outputs.albedo, "cesm2band", flx_slr=outputs.flx_slr)

# 3. Access band albedos and indices
print(s2.B3)        # green-band albedo
print(s2.NDSI)      # Normalized Difference Snow Index
print(cesm.vis)     # VIS broadband albedo
print(cesm.nir)     # NIR broadband albedo

print(f"Broadband albedo: {outputs.BBA}")
