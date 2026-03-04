"""Inverse modelling for BioSNICAR: MLP emulator + spectral retrieval.

Provides a fast neural-network emulator of the BioSNICAR forward model and
optimisation routines to retrieve ice physical properties from observed
spectral or satellite-band albedo.

Quick start
-----------
>>> from biosnicar.inverse import Emulator, retrieve
>>> emu = Emulator.load("glacier_ice.npz")
>>> result = retrieve(observed=obs, parameters=["rds", "rho"], emulator=emu)
>>> print(result.summary())
"""

from biosnicar.inverse.emulator import Emulator
from biosnicar.inverse.optimize import retrieve, DEFAULT_BOUNDS, DEFAULT_X0
from biosnicar.inverse.result import RetrievalResult

__all__ = [
    "Emulator",
    "retrieve",
    "RetrievalResult",
    "DEFAULT_BOUNDS",
    "DEFAULT_X0",
]
