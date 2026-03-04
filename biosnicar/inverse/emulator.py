"""Backward-compatible re-export.

The canonical location for the Emulator class is now
:mod:`biosnicar.emulator`.  This module re-exports it so that
``from biosnicar.inverse.emulator import Emulator`` continues to work.
"""

from biosnicar.emulator import Emulator, _latin_hypercube, _snap_rds  # noqa: F401
